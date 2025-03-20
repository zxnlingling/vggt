import pandas as pd

def subtract_consecutive_rows(csv_file, output_file):
    # Read the csv file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if the DataFrame has at least two rows
    if len(df) < 2:
        print("The CSV file must have at least two rows.")
        return
    
    # Subtract consecutive rows
    result_df = df.diff().iloc[1:]
    
    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)
    print(f"Result saved to {output_file}")

# Example usage
csv_file_path = 'camera_poses2.csv'
output_file_path = 'camera_poses_sub2.csv'
subtract_consecutive_rows(csv_file_path, output_file_path)