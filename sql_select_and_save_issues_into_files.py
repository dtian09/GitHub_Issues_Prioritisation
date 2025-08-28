import mysql.connector
import subprocess

# List of issue IDs to export
'''
issue_ids = [1148298434, # Shortest issue
            830038177,
            58861063, 
            591940611,
            916830210,
            236489478,
            381847808,
            705332485,
            58498821,
            1097457670]
'''

issue_ids = [ #longest issues
    941150350,
    354702553,
    237734712,
    315565490
    ]

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="david",
    password="david",
    database="github_issues_db"
)
cursor = conn.cursor()
#destination_dir = "/mnt/c/users/dtian/GitHub_Issues_Prioritisation/shortest_issues/"
destination_dir = "/mnt/c/users/dtian/GitHub_Issues_Prioritisation/longest_issues/"

for issue_id in issue_ids:
    # Define output file path (must match secure_file_priv dir)
    outfile_path = f"/var/lib/mysql-files/issue_id_{issue_id}.txt"

    # MySQL INTO OUTFILE query (no header, single row)
    query = f"""
    SELECT content
    FROM issue
    WHERE issue_id = {issue_id}
    INTO OUTFILE '{outfile_path}'
    FIELDS TERMINATED BY ''
    LINES TERMINATED BY '\n';
    """

    try:
        cursor.execute(query)
        print(f"✅ Exported issue_id {issue_id} to {outfile_path}")        
    except mysql.connector.Error as err:
        print(f"❌ Failed to export issue_id {issue_id}: {err}")   
    try:
        # Move file to destination using sudo mv
        subprocess.run(["sudo", "mv", outfile_path, destination_dir], check=True)
        print(f"📁 Moved to {destination_dir}")
    except subprocess.CalledProcessError as err:
        print(f"❌ Failed to move issue_id {issue_id}: {err}")

cursor.close()
conn.close()
