import os
import math
import pickle
import logging
import collections
import numpy as np
from natsort import natsort
import multiprocessing as mp
from multiprocessing import Lock
from abc import ABC, abstractmethod
from typing import Tuple, List, Type, Any
from torch.utils.data import IterableDataset, DataLoader


# String constants for storage
ACTION = 'action'
REWARD = 'reward'
TERMINAL = 'terminal'
TIMEOUT = 'timeout'
INDICES = 'indices'


def invalid_range(cursor, replay_capacity, stack_size, update_horizon):
    """Returns a array with the indices of cursor-related invalid transitions.

    There are update_horizon + stack_size invalid indices:
      - The update_horizon indices before the cursor, because we do not have a
        valid N-step transition (including the next state).
      - The stack_size indices on or immediately after the cursor.
    If N = update_horizon, K = stack_size, and the cursor is at c, invalid
    indices are:
      c - N, c - N + 1, ..., c, c + 1, ..., c + K - 1.

    It handles special cases in a circular buffer in the beginning and the end.

    Args:
      cursor: int, the position of the cursor.
      replay_capacity: int, the size of the replay memory.
      stack_size: int, the size of the stacks returned by the replay memory.
      update_horizon: int, the agent's update horizon.
    Returns:
      np.array of size stack_size with the invalid indices.
    """
    assert cursor < replay_capacity
    return np.array(
        [(cursor - update_horizon + i) % replay_capacity
         for i in range(stack_size + update_horizon)])


class ObservationElement(object):

    def __init__(self, name: str, shape: tuple, type: Type[np.dtype]):
        self.name = name
        self.shape = shape
        self.type = type


class ReplayElement:
    def __init__(self, name, shape, type, is_observation=False):
        self.name, self.shape, self.type, self.is_observation = name, shape, type, is_observation


class ReplayBuffer(ABC):
    def replay_capacity(self): pass
    def batch_size(self): pass
    def get_storage_signature(self) -> Tuple[List[ReplayElement], List[ReplayElement]]: pass
    def add(self, action, reward, terminal, timeout, **kwargs): pass
    def add_final(self, **kwargs): pass
    def is_empty(self): pass
    def is_full(self): pass
    def cursor(self): pass
    def set_cursor(self): pass
    def get_range(self, array, start_index, end_index): pass
    def get_range_stack(self, array, start_index, end_index, terminals=None): pass
    def get_terminal_stack(self, index): pass
    def is_valid_transition(self, index): pass
    def sample_index_batch(self, batch_size): pass
    def unpack_transition(self, transition_tensors, transition_type): pass
    def sample_transition_batch(self, batch_size=None, indices=None, pack_in_dict=True): pass
    def get_transition_elements(self, batch_size=None): pass
    def shutdown(self): pass
    def using_disk(self): pass


class UniformReplayBuffer(ReplayBuffer):
    """A simple out-of-graph Replay Buffer.

    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.

    When the states consist of stacks of observations storing the states is
    inefficient. This class writes observations and constructs the stacked states
    at sample time.

    Attributes:
      _add_count: int, counter of how many transitions have been added (including
        the blank ones at the beginning of an episode).
      invalid_range: np.array, an array with the indices of cursor-related invalid
        transitions
    """

    def __init__(self,
                 batch_size: int = 32,
                 timesteps: int = 1,
                 replay_capacity: int = int(1e6),
                 update_horizon: int = 1,
                 gamma: float = 0.99,
                 max_sample_attempts: int = 10000,
                 action_shape: tuple = (),
                 action_dtype: Type[np.dtype] = np.float32,
                 reward_shape: tuple = (),
                 reward_dtype: Type[np.dtype] = np.float32,
                 observation_elements: List[ObservationElement] = None,
                 extra_replay_elements: List[ReplayElement] = None,
                 disk_saving: bool = False,
                 purge_replay_on_shutdown: bool = True
                 ):
        """Initializes OutOfGraphReplayBuffer.

        Args:
          batch_size: int.
          timesteps: int, number of frames to use in state stack.
          replay_capacity: int, number of transitions to keep in memory.
          update_horizon: int, length of update ('n' in n-step update).
          # gamma: int, the discount factor.
          # max_sample_attempts: int, the maximum number of attempts allowed to
            get a sample.
          action_shape: tuple of ints, the shape for the action vector.
            Empty tuple means the action is a scalar.
          action_dtype: np.dtype, type of elements in the action.
          reward_shape: tuple of ints, the shape of the reward vector.
            Empty tuple means the reward is a scalar.
          reward_dtype: np.dtype, type of elements in the reward.
          observation_elements: list of ObservationElement defining the type of
            the extra contents that will be stored and returned.
          extra_storage_elements: list of ReplayElement defining the type of
            the extra contents that will be stored and returned.
          # purge_replay_on_shutdown

        Raises:
          ValueError: If replay_capacity is too small to hold at least one
            transition.
        """

        if observation_elements is None:
            observation_elements = []
        if extra_replay_elements is None:
            extra_replay_elements = []

        if replay_capacity < update_horizon + timesteps:
            raise ValueError('There is not enough capacity to cover '
                             'update_horizon and stack_size.')

        logging.info(
            'Creating a %s replay memory with the following parameters:',
            self.__class__.__name__)
        logging.info('\t timesteps: %d', timesteps)
        logging.info('\t replay_capacity: %d', replay_capacity)
        logging.info('\t batch_size: %d', batch_size)
        logging.info('\t update_horizon: %d', update_horizon)
        logging.info('\t gamma: %f', gamma)

        self._disk_saving = disk_saving
        self._purge_replay_on_shutdown = purge_replay_on_shutdown

        if not self._disk_saving:
            logging.info('\t saving to RAM')


        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._timesteps = timesteps
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._max_sample_attempts = max_sample_attempts

        self._observation_elements = observation_elements
        self._extra_replay_elements = extra_replay_elements

        self._storage_signature, self._obs_signature = self.get_storage_signature()
        self._create_storage()

        self._lock = Lock()
        self._add_count = mp.Value('i', 0)

        self.invalid_range = np.zeros((self._timesteps))

        self._valid_sample_indices = None
        self._enumeration_cursor = None

        # When the horizon is > 1, we compute the sum of discounted rewards as a dot
        # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(update_horizon)],
            dtype=np.float32)

        # store a global mapping from global index to the index of each individual task
        self._index_mapping = np.full((self._replay_capacity, 2), -1)
        self._task_names = []
        self._task_replay_storage_folders = []
        self._task_add_count = []
        self._task_replay_start_index = []
        self._task_index = {}
        self._num_tasks = 0

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def replay_capacity(self):
        return self._replay_capacity

    @property
    def batch_size(self):
        return self._batch_size

    def _create_storage(self, store=None):
        """Creates the numpy arrays used to store transitions.
        """
        self._store = {} if store is None else store
        for storage_element in self._storage_signature:
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            if storage_element.name == TERMINAL:
                self._store[storage_element.name] = np.full(
                    array_shape, -1, dtype=storage_element.type)
            elif not self._disk_saving:
                # If saving to disk, we don't need to store anything else.
                self._store[storage_element.name] = np.empty(
                    array_shape, dtype=storage_element.type)

    def get_storage_signature(self) -> Tuple[List[ReplayElement],
                                             List[ReplayElement]]:
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
          dict of ReplayElements defining the type of the contents stored.
        """
        storage_elements = [
            ReplayElement(ACTION, self._action_shape, self._action_dtype),
            ReplayElement(REWARD, self._reward_shape, self._reward_dtype),
            ReplayElement(TERMINAL, (), np.int8),
            ReplayElement(TIMEOUT, (), np.bool),
        ]

        obs_elements = []
        for obs_element in self._observation_elements:
            obs_elements.append(
                ReplayElement(
                    obs_element.name, obs_element.shape, obs_element.type))
        storage_elements.extend(obs_elements)

        for extra_replay_element in self._extra_replay_elements:
            storage_elements.append(extra_replay_element)

        return storage_elements, obs_elements

    def add(self, task, task_replay_storage_folder, action, reward, terminal, timeout, **kwargs):
        """Adds a transition to the replay memory.

        WE ONLY STORE THE TPS1s on the final frame

        This function checks the types and handles the padding at the beginning of
        an episode. Then it calls the _add function.

        Since the next_observation in the transition will be the observation added
        next there is no need to pass it.

        If the replay memory is at capacity the oldest transition will be discarded.

        Args:
          action: int, the action in the transition.
          reward: float, the reward received in the transition.
          terminal: A uint8 acting as a boolean indicating whether the transition
                    was terminal (1) or not (0).
          **kwargs: The remaining args
        """

        # If previous transition was a terminal, then add_final wasn't called
        # if not self.is_empty() and self._store['terminal'][self.cursor() - 1] == 1:
        #     raise ValueError('The previous transition was a terminal, '
        #                      'but add_final was not called.')

        kwargs[ACTION] = action
        kwargs[REWARD] = reward
        kwargs[TERMINAL] = terminal
        kwargs[TIMEOUT] = timeout
        self._check_add_types(kwargs, self._storage_signature)
        self._add(task, task_replay_storage_folder, kwargs)

    def recover_from_disk(self, task, task_replay_storage_folder):
        if os.path.exists(os.path.join(task_replay_storage_folder, 'replay_info.npy')):
            with open(os.path.join(task_replay_storage_folder, 'replay_info.npy'), 'rb') as fp:
                replay_info = np.load(fp)
        else:
            replay_indices = [int(filename.split('.')[0]) for filename in os.listdir(task_replay_storage_folder) 
                              if filename.endswith('.replay')]
            replay_indices.sort()
            replay_info = np.zeros(len(replay_indices), dtype = np.int8)
            for i, index in enumerate(replay_indices):
                with self._lock:
                    with open(os.path.join(task_replay_storage_folder, '{}.replay'.format(index)), 'rb') as fp:
                        kwargs = pickle.load(fp)
                    replay_info[i] = kwargs[TERMINAL]
            with open(os.path.join(task_replay_storage_folder, 'replay_info.npy'), 'wb') as fp:
                np.save(fp, replay_info)

        if task not in self._task_index:
            self._task_names.append(task)
            self._task_replay_storage_folders.append(task_replay_storage_folder)
            self._task_index[task] = len(self._task_replay_storage_folders) - 1 
            # NOTE: need to guarantee that the replays from the same task are loaded consecutively
            self._task_replay_start_index.append(self.cursor())
            self._num_tasks += 1
            self._task_add_count.append(mp.Value('i', 0))

        task_idx = self._task_index[task]

        for i in range(len(replay_info)):
            cursor = self.cursor()
            self._store[TERMINAL][cursor] = replay_info[i]

            self._index_mapping[cursor, 0] = task_idx
            self._index_mapping[cursor, 1] = i

            with self._task_add_count[task_idx].get_lock():
                self._task_add_count[task_idx].value += 1

            with self._add_count.get_lock():
                self._add_count.value += 1

            self.invalid_range = invalid_range(
                self.cursor(), self._replay_capacity, self._timesteps,
                self._update_horizon)

    def add_final(self, task, task_replay_storage_folder, **kwargs):
        """Adds a transition to the replay memory.
        Args:
          **kwargs: The remaining args
        """
        # if self.is_empty() or self._store['terminal'][self.cursor() - 1] != 1:
        #     raise ValueError('The previous transition was not terminal.')
        self._check_add_types(kwargs, self._obs_signature)
        transition = self._final_transition(kwargs)
        self._add(task, task_replay_storage_folder, transition)

    def _final_transition(self, kwargs):
        transition = {}
        for element_type in self._storage_signature:
            if element_type.name in kwargs:
                transition[element_type.name] = kwargs[element_type.name]
            elif element_type.name == TERMINAL:
                # Used to check that user is correctly adding transitions
                transition[element_type.name] = -1
            else:
                transition[element_type.name] = np.empty(
                    element_type.shape, dtype=element_type.type)
        return transition

    def _add_initial_to_disk(self, kwargs: dict):
        for i in range(self._timesteps - 1):
            with open(os.path.join(kwargs['task_replay_storage_folder'], '%d.replay' % (
                    self._replay_capacity - 1 - i)), 'wb') as f:
                pickle.dump(kwargs, f)

    def _add(self, task, task_replay_storage_folder: str, kwargs: dict):
        """Internal add method to add to the storage arrays.

        Args:
          kwargs: All the elements in a transition.
        """
        with self._lock:
            cursor = self.cursor()

            if self._disk_saving:
                assert task_replay_storage_folder is not None

                term = self._store[TERMINAL]
                term[cursor] = kwargs[TERMINAL]
                self._store[TERMINAL] = term
                # self._store[TERMINAL][cursor] = kwargs[TERMINAL]

                # create mapping from global index to task index
                if task not in self._task_index:
                    self._task_names.append(task)
                    self._task_replay_storage_folders.append(task_replay_storage_folder)
                    self._task_index[task] = len(self._task_replay_storage_folders) - 1
                    self._task_replay_start_index.append(cursor) 
                    # NOTE: need to guarantee that the replays from the same task are loaded consecutively
                    self._num_tasks += 1
                    self._task_add_count.append(mp.Value('i', 0))

                task_idx = self._task_index[task]
                task_cursor = self._task_add_count[task_idx].value

                self._index_mapping[cursor, 0] = task_idx
                self._index_mapping[cursor, 1] = task_cursor

                with open(os.path.join(task_replay_storage_folder, '%d.replay' % task_cursor), 'wb') as f:
                    pickle.dump(kwargs, f)
                # If first add, then pad for correct wrapping
                # TODO: it's gonna be wrong if self.timestep > 1, no mapping for initial steps 
                # (which are indexed as the end of the buffer)
                if self._add_count.value == 0:
                    self._add_initial_to_disk(kwargs)

                with self._task_add_count[task_idx].get_lock():
                    self._task_add_count[task_idx].value += 1
            else:
                for name, data in kwargs.items():
                    item = self._store[name]
                    item[cursor] = data
                    self._store[name] = item

                    # self._store[name][cursor] = data

            with self._add_count.get_lock():
                self._add_count.value += 1
            self.invalid_range = invalid_range(
                self.cursor(), self._replay_capacity, self._timesteps,
                self._update_horizon)

    def _get_from_disk(self, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Here we fake a mini store (buffer)
        store = {store_element.name: {}
                 for store_element in self._storage_signature}
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            for i in range(start_index, end_index):
                task_replay_storage_folder = self._task_replay_storage_folders[self._index_mapping[i, 0]]
                task_index = self._index_mapping[i, 1]

                with open(os.path.join(task_replay_storage_folder, '%d.replay' % task_index), 'rb') as f:
                    d = pickle.load(f)
                    for k, v in d.items():
                        ###### new ######
                        if k not in store:
                            store[k] = [None] * self._replay_capacity
                        ###### new ######
                        store[k][i] = v # NOTE: potential bug here, should % self._replay_capacity
        else:
            for i in range(end_index - start_index):
                idx = (start_index + i) % self._replay_capacity
                task_replay_storage_folder = self._task_replay_storage_folders[self._index_mapping[idx, 0]]
                task_index = self._index_mapping[idx, 1]
                with open(os.path.join(task_replay_storage_folder, '%d.replay' % task_index), 'rb') as f:
                    d = pickle.load(f)
                    for k, v in d.items():
                        ###### new ######
                        if k not in store:
                            store[k] = [None] * self._replay_capacity
                        ###### new ######
                        store[k][idx] = v
        return store

    def _check_add_types(self, kwargs, signature):
        """Checks if args passed to the add method match those of the storage.

        Args:
          *args: Args whose types need to be validated.

        Raises:
          ValueError: If args have wrong shape or dtype.
        """

        if (len(kwargs)) != len(signature):
            expected = str(natsort.natsorted([e.name for e in signature]))
            actual = str(natsort.natsorted(list(kwargs.keys())))
            error_list = '\nList of expected:\n{}\nList of actual:\n{}'.format(
                expected, actual)
            raise ValueError('Add expects {} elements, received {}.'.format(
                len(signature), len(kwargs)) + error_list)

        for store_element in signature:
            arg_element = kwargs[store_element.name]
            if isinstance(arg_element, np.ndarray):
                arg_shape = arg_element.shape
            elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
                # TODO: This is not efficient when arg_element is a list.
                arg_shape = np.array(arg_element).shape
            else:
                # Assume it is scalar.
                arg_shape = tuple()
            store_element_shape = tuple(store_element.shape)
            if arg_shape != store_element_shape:
                import pdb;pdb.set_trace()
                raise ValueError('arg has shape {}, expected {}'.format(
                    arg_shape, store_element_shape))

    def is_empty(self):
        """Is the Replay Buffer empty?"""
        return self._add_count.value == 0

    def is_full(self):
        """Is the Replay Buffer full?"""
        return self._add_count.value >= self._replay_capacity

    def cursor(self):
        """Index to the location where the next transition will be written."""
        return self._add_count.value % self._replay_capacity

    @property
    def add_count(self):
        return np.array(self._add_count.value) #self._add_count.copy()

    @add_count.setter
    def add_count(self, count):
        if isinstance(count, int):
            self._add_count = mp.Value('i', count)
        else:
            self._add_count = count


    def get_range(self, array, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          array: np.array, the array to get the stack from.
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Fast slice read when there is no wraparound.
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            return_array = np.array(
                [array[i] for i in range(start_index, end_index)])
        # Slow list read.
        else:
            indices = [(start_index + i) % self._replay_capacity
                       for i in range(end_index - start_index)]
            return_array = np.array([array[i] for i in indices])

        return return_array

    def get_range_stack(self, array, start_index, end_index, terminals=None):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          array: np.array, the array to get the stack from.
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        return_array = np.array(self.get_range(array, start_index, end_index))
        if terminals is None:
            terminals = self.get_range(
                self._store[TERMINAL], start_index, end_index)

        terminals = terminals[:-1]

        # Here we now check if we need to pad the front episodes
        # If any have a terminal of -1, then we have spilled over
        # into the the previous transition
        if np.any(terminals == -1):
            padding_item = return_array[-1]
            _array = list(return_array)[:-1]
            arr_len = len(_array)
            pad_from_now = False
            for i, (ar, term) in enumerate(
                    zip(reversed(_array), reversed(terminals))):
                if term == -1 or pad_from_now:
                    # The first time we see a -1 term, means we have hit the
                    # beginning of this episode, so pad from now.
                    # pad_from_now needed because the next transition (reverse)
                    # will not be a -1 terminal.
                    pad_from_now = True
                    return_array[arr_len - 1 - i] = padding_item
                else:
                    # After we hit out first -1 terminal, we never reassign.
                    padding_item = ar

        return return_array

    def _get_element_stack(self, array, index, terminals=None):
        state = self.get_range_stack(array,
                                     index - self._timesteps + 1, index + 1,
                                     terminals=terminals)
        return state

    def get_terminal_stack(self, index):
        terminal_stack = self.get_range(self._store[TERMINAL],
                              index - self._timesteps + 1,
                              index + 1)
        return terminal_stack

    def is_valid_transition(self, index):
        """Checks if the index contains a valid transition.

        Checks for collisions with the end of episodes and the current position
        of the cursor.

        Args:
          index: int, the index to the state in the transition.

        Returns:
          Is the index valid: Boolean.

        """
        # Check the index is in the valid range
        if index < 0 or index >= self._replay_capacity:
            return False
        if not self.is_full():
            # The indices and next_indices must be smaller than the cursor.
            if index >= self.cursor() - self._update_horizon:
                return False

        # Skip transitions that straddle the cursor.
        if index in set(self.invalid_range):
            return False

        term_stack = self.get_terminal_stack(index)
        if term_stack[-1] == -1:
            return False

        return True

    def _create_batch_arrays(self, batch_size):
        """Create a tuple of arrays with the type of get_transition_elements.

        When using the WrappedReplayBuffer with staging enabled it is important
        to create new arrays every sample because StaginArea keeps a pointer to
        the returned arrays.

        Args:
          batch_size: (int) number of transitions returned. If None the default
            batch_size will be used.

        Returns:
          Tuple of np.arrays with the shape and type of get_transition_elements.
        """
        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = []
        for element in transition_elements:
            batch_arrays.append(np.empty(element.shape, dtype=element.type))
        return tuple(batch_arrays)

    def sample_index_batch(self, batch_size, distribution_mode = 'transition_uniform'):
        """Returns a batch of valid indices sampled uniformly.

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly.

        Raises:
          RuntimeError: If the batch was not constructed after maximum number of
            tries.
        """
        # print('distribution_mode: ', distribution_mode)
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = (self.cursor() - self._replay_capacity +
                      self._timesteps - 1)
            max_id = self.cursor() - self._update_horizon
        else:
            min_id = 0
            max_id = self.cursor() - self._update_horizon
            if max_id <= min_id:
                raise RuntimeError(
                    'Cannot sample a batch with fewer than stack size '
                    '({}) + update_horizon ({}) transitions.'.
                    format(self._timesteps, self._update_horizon))

        indices = []
        if distribution_mode == 'transition_uniform':
            attempt_count = 0
            while (len(indices) < batch_size and
                        attempt_count < self._max_sample_attempts):
                index = np.random.randint(min_id, max_id) % self._replay_capacity
                if self.is_valid_transition(index):
                    indices.append(index)
                else:
                    attempt_count += 1
        elif distribution_mode == 'task_uniform':
            task_indices = np.random.randint(low = 0, high = self._num_tasks, size = batch_size)
            for task_index in task_indices:
                attempt_count = 0
                while attempt_count < self._max_sample_attempts:
                    state_index = np.random.randint(low = self._task_replay_start_index[task_index], \
                                                    high = self._task_replay_start_index[task_index] \
                                                        + self._task_add_count[task_index].value)
                    if self.is_valid_transition(state_index):
                        indices.append(state_index)
                        break
                    else:
                        attempt_count += 1
        else:
            raise NotImplementedError

        if len(indices) != batch_size:
            raise RuntimeError(
                'Max sample attempts: Tried {} times but only sampled {}'
                ' valid indices. Batch size is {}'.
                    format(self._max_sample_attempts, len(indices), batch_size))

        return indices

    def unpack_transition(self, transition_tensors, transition_type):
        """Unpacks the given transition into member variables.

        Args:
          transition_tensors: tuple of tf.Tensors.
          transition_type: tuple of ReplayElements matching transition_tensors.
        """
        self.transition = collections.OrderedDict()
        for element, element_type in zip(transition_tensors, transition_type):
            self.transition[element_type.name] = element
        return self.transition

    def sample_transition_batch(self, batch_size=None, indices=None,
                                pack_in_dict=True, distribution_mode = 'transition_uniform'):
        """Returns a batch of transitions (including any extra contents).

        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.

        When the transition is terminal next_state_batch has undefined contents.

        NOTE: This transition contains the indices of the sampled elements.
        These are only valid during the call to sample_transition_batch,
        i.e. they may  be used by subclasses of this replay buffer but may
        point to different data as soon as sampling is done.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().

        Raises:
          ValueError: If an element to be sampled is missing from the
            replay buffer.
        """

        if batch_size is None:
            batch_size = self._batch_size
        with self._lock:
            if indices is None:
                indices = self.sample_index_batch(batch_size, distribution_mode)
            assert len(indices) == batch_size

            transition_elements = self.get_transition_elements(batch_size)
            batch_arrays = self._create_batch_arrays(batch_size)
            task_name_arrays = []

            for batch_element, state_index in enumerate(indices):

                if not self.is_valid_transition(state_index):
                    raise ValueError('Invalid index %d.' % state_index)

                task_name_arrays.append(self._task_names[self._index_mapping[state_index, 0]])

                trajectory_indices = [(state_index + j) % self._replay_capacity
                                      for j in range(self._update_horizon)]
                trajectory_terminals = self._store['terminal'][
                    trajectory_indices]
                is_terminal_transition = trajectory_terminals.any()
                if not is_terminal_transition:
                    trajectory_length = self._update_horizon
                else:
                    # np.argmax of a bool array returns index of the first True.
                    trajectory_length = np.argmax(
                        trajectory_terminals.astype(np.bool),
                        0) + 1

                next_state_index = state_index + trajectory_length

                store = self._store
                if self._disk_saving:
                    store = self._get_from_disk(
                        state_index - (self._timesteps - 1),
                        next_state_index + 1)

                trajectory_discount_vector = (
                    self._cumulative_discount_vector[:trajectory_length])
                trajectory_rewards = self.get_range(store['reward'],
                                                    state_index,
                                                    next_state_index)

                terminal_stack = self.get_terminal_stack(state_index)
                terminal_stack_tp1 = self.get_terminal_stack(
                    next_state_index % self._replay_capacity)

                # Fill the contents of each array in the sampled batch.
                assert len(transition_elements) == len(batch_arrays)
                for element_array, element in zip(batch_arrays,
                                                  transition_elements):
                    if element.is_observation:
                        if element.name.endswith('tp1'):
                            element_array[
                                batch_element] = self._get_element_stack(
                                store[element.name[:-4]],
                                next_state_index % self._replay_capacity,
                                terminal_stack_tp1)
                        else:
                            element_array[
                                batch_element] = self._get_element_stack(
                                store[element.name],
                                state_index, terminal_stack)
                    elif element.name == REWARD:
                        # compute discounted sum of rewards in the trajectory.
                        element_array[batch_element] = np.sum(
                            trajectory_discount_vector * trajectory_rewards,
                            axis=0)
                    elif element.name == TERMINAL:
                        element_array[batch_element] = is_terminal_transition
                    elif element.name == INDICES:
                        element_array[batch_element] = state_index
                    elif element.name in store.keys():
                        try:
                            element_array[batch_element] = (
                                store[element.name][state_index])
                        except:
                            import IPython
                            IPython.embed()

        if pack_in_dict:
            batch_arrays = self.unpack_transition(
                batch_arrays, transition_elements)

        # TODO: make a proper fix for this
        if 'task' in batch_arrays:
            del batch_arrays['task']
        if 'task_tp1' in batch_arrays:
            del batch_arrays['task_tp1']

        batch_arrays['tasks'] = task_name_arrays

        return batch_arrays

    def enumerate_next_transition_batch(self, batch_size=None, pack_in_dict=True):
        '''
        the last batch will be kept
        '''
        if batch_size is None:
            batch_size = self._batch_size
        with self._lock:
            indices = []
            for _ in range(batch_size):
                indices.append(self._valid_sample_indices[self._enumeration_cursor])
                self._enumeration_cursor = self._enumeration_cursor + 1
                if self._enumeration_cursor >= len(self._valid_sample_indices):
                    self._enumeration_cursor = 0
                    break
            batch_size = len(indices)
        return self.sample_transition_batch(batch_size, indices, pack_in_dict)

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [
            ReplayElement(ACTION, (batch_size,) + self._action_shape,
                          self._action_dtype),
            ReplayElement(REWARD, (batch_size,) + self._reward_shape,
                          self._reward_dtype),
            ReplayElement(TERMINAL, (batch_size,), np.int8),
            ReplayElement(TIMEOUT, (batch_size,), np.bool),
            ReplayElement(INDICES, (batch_size,), np.int32),
        ]

        for element in self._observation_elements:
            transition_elements.append(ReplayElement(
                element.name,
                (batch_size, self._timesteps) + tuple(element.shape),
                element.type, True))
            transition_elements.append(ReplayElement(
                element.name + '_tp1',
                (batch_size, self._timesteps) + tuple(element.shape),
                element.type, True))

        for element in self._extra_replay_elements:
            transition_elements.append(ReplayElement(
                element.name,
                (batch_size,) + tuple(element.shape),
                element.type))
        return transition_elements

    def shutdown(self):
        if self._purge_replay_on_shutdown:
            # Safely delete replay
            logging.info('Clearing disk replay buffer.')
            for f in [f for f in os.listdir(self._save_dir) if '.replay' in f]:
                os.remove(os.path.join(self._save_dir, f))

    def using_disk(self):
        return self._disk_saving

    def prepare_enumeration(self):
        '''
        get the number of indices that can be sampled
        '''
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = (self.cursor() - self._replay_capacity +
                      self._timesteps - 1)
            max_id = self.cursor() - self._update_horizon
        else:
            min_id = 0
            max_id = self.cursor() - self._update_horizon
            if max_id <= min_id:
                raise RuntimeError(
                    'Cannot sample a batch with fewer than stack size '
                    '({}) + update_horizon ({}) transitions.'.
                    format(self._timesteps, self._update_horizon))

        self._valid_sample_indices = []
        for raw_idx in range(min_id, max_id):
            index = raw_idx % self._replay_capacity
            if self.is_valid_transition(index):
                self._valid_sample_indices.append(index)

        self._enumeration_cursor = 0

        return len(self._valid_sample_indices)

    def init_enumeratation(self):
        self._enumeration_cursor = 0


class WrappedReplayBuffer(ABC):
    def __init__(self, replay_buffer: ReplayBuffer):
        self._replay_buffer = replay_buffer

    @property
    def replay_buffer(self): return self._replay_buffer

    @abstractmethod
    def dataset(self) -> Any: pass


class PyTorchIterableReplayDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer, sample_mode, sample_distribution_mode='transition_uniform'):
        self._replay_buffer, self._sample_mode, self._sample_distribution_mode = \
            replay_buffer, sample_mode, sample_distribution_mode
        if self._sample_mode == 'enumerate': self._num_samples = self._replay_buffer.prepare_enumeration()

    def _generator(self):
        while True:
            yield self._replay_buffer.sample_transition_batch(pack_in_dict=True, \
                                                              distribution_mode=self._sample_distribution_mode) \
                                                                if self._sample_mode == 'random' \
                                                                    else self._replay_buffer.enumerate_next_transition_batch(
                                                                        pack_in_dict=True)

    def __iter__(self): return iter(self._generator())
    def __len__(self): return self._num_samples // self._replay_buffer._batch_size


class PyTorchReplayBuffer(WrappedReplayBuffer):
    def __init__(self, replay_buffer: ReplayBuffer, num_workers: int = 2, sample_mode='random', \
                 sample_distribution_mode='transition_uniform'):
        super().__init__(replay_buffer)
        self._num_workers, self._sample_mode, self._sample_distribution_mode = num_workers, sample_mode, \
            sample_distribution_mode

    def dataset(self) -> DataLoader:
        return DataLoader(PyTorchIterableReplayDataset(self._replay_buffer, self._sample_mode, \
                                                       self._sample_distribution_mode), batch_size=None, \
                                                        pin_memory=True, num_workers=self._num_workers)