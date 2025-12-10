# =========================================
# 📊 Workshop Attendance Analysis (University-based Version)
# =========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------
# 1️⃣ Base Setup
# -----------------------------------------
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)

# File Paths (Update these as needed)
# You can add multiple file paths to this list
attendance_file_paths = [
    '/content/2023_2024_2025__Attendance_Report (1).xlsx',
    '/content/Another_Attendance_Report.csv'
]
taxonomy_file_path = '/content/Taxonomy.csv'  # Assumed path for taxonomy

# -----------------------------------------
# 2️⃣ Load Data
# -----------------------------------------
# -----------------------------------------
# 2️⃣ Load Data
# -----------------------------------------
def load_data():
    # Load Attendance Data
    df_list = []
    files_found = False
    
    for path in attendance_file_paths:
        if os.path.exists(path):
            try:
                if path.lower().endswith('.csv'):
                    df_temp = pd.read_csv(path)
                else:
                    df_temp = pd.read_excel(path)
                df_list.append(df_temp)
                print(f"✅ Loaded: {path}")
                files_found = True
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
        else:
            print(f"⚠️ File not found: {path}")

    if files_found and df_list:
        df_att = pd.concat(df_list, ignore_index=True)
        print(f"✅ Combined {len(df_list)} attendance files. Total records: {len(df_att)}")
    else:
        print("⚠️ No valid attendance data found. Please check your file paths.")
        df_att = pd.DataFrame() # Return empty DataFrame

    # Load Taxonomy Data
    try:
        if os.path.exists(taxonomy_file_path):
            if taxonomy_file_path.lower().endswith('.csv'):
                df_tax = pd.read_csv(taxonomy_file_path)
            else:
                df_tax = pd.read_excel(taxonomy_file_path)
            print("✅ Taxonomy data loaded.")
        else:
            print(f"⚠️ Taxonomy file not found: {taxonomy_file_path}")
            df_tax = pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading taxonomy: {e}")
        df_tax = pd.DataFrame()
    
    return df_att, df_tax

df, df_taxonomy = load_data()

if df.empty or df_taxonomy.empty:
    print("❌ Insufficient data to proceed. Please ensure both Attendance and Taxonomy files are present and contain data.")
    exit()

# -----------------------------------------
# 3️⃣ Data Cleaning & Merging
# -----------------------------------------
# Clean Attendance Columns
# We expect specific columns based on user input:
# 'Event Name', 'University Program', 'Attendance Status', etc.
df.columns = df.columns.str.strip() # Just strip whitespace, don't lower/replace yet to preserve original mapping if needed, but standardizing is safer.
# Let's standardize to snake_case for internal use but map from the known attributes.
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)

# Map known attributes to internal names
rename_map = {
    'simid': 'student_id', 
    'university_program': 'university_program',
    'attendance_status': 'attendance_status',
    'event_name': 'event_name'
}
df = df.rename(columns=rename_map)

# Clean Event Names in both datasets for matching
if 'event_name' in df.columns:
    df['event_name_clean'] = df['event_name'].str.replace(r'[\(\)]', '', regex=True).str.strip().str.lower()
else:
    print("❌ Critical Error: 'Event Name' column not found in attendance data.")
    exit()

if 'Career Development Workshop Titles' in df_taxonomy.columns:
    df_taxonomy['workshop_title_clean'] = df_taxonomy['Career Development Workshop Titles'].str.replace(r'[\(\)]', '', regex=True).str.strip().str.lower()
else:
    # Try to find it if column cleaning happened or if name varies
    # But user said "stick to all the goven attributes", so we expect 'Career Development Workshop Titles'
    print("❌ Critical Error: 'Career Development Workshop Titles' column not found in taxonomy.")
    # Check if we should try to clean taxonomy columns too? 
    # Let's assume taxonomy file has headers as described.
    pass 


# Merge Attendance with Taxonomy
# Left join on cleaned event names
df = df.merge(
    df_taxonomy[['workshop_title_clean', 'Category', 'Sub-Category']], 
    left_on='event_name_clean', 
    right_on='workshop_title_clean', 
    how='left'
)

# Fill missing taxonomy
df['Category'] = df['Category'].fillna('Uncategorized')
df['Sub-Category'] = df['Sub-Category'].fillna('Uncategorized')

# Attendance Flag
df['is_attended'] = df['attendance_status'].apply(lambda x: 1 if str(x).lower() == 'attended' else 0)

# University Category
# Check if university_program exists, if not try to find a fallback or warn
if 'university_program' not in df.columns:
    print(f"⚠️ Columns found: {df.columns.tolist()}")
    # Try to find a column that looks like it
    possible_cols = [c for c in df.columns if 'university' in c or 'program' in c]
    if possible_cols:
        print(f"⚠️ Using {possible_cols[0]} as university_program")
        df['university_program'] = df[possible_cols[0]]
    else:
        df['university_program'] = 'Unknown'

df['uni_category'] = df['university_program'].fillna('Unknown')

# Filter Attended Only
df_attended = df[df['is_attended'] == 1].copy()

# -----------------------------------------
# 4️⃣ Color Mapping
# -----------------------------------------
all_universities = sorted(df['uni_category'].unique())
color_palette = plt.cm.tab20.colors
university_colors = {uni: color_palette[i % len(color_palette)] for i, uni in enumerate(all_universities)}

# -----------------------------------------
# 5️⃣ Graph Generation Functions
# -----------------------------------------

def plot_stacked_bar(df_data, title, filename_suffix):
    if df_data.empty:
        print(f"Skipping {title}: No data.")
        return

    # Pivot: Index=Event, Columns=Uni, Values=Count
    df_pivot = df_data.pivot(index='event_name', columns='uni_category', values='is_attended').fillna(0)
    
    # ---------------------------------------------------------
    # SORTING LOGIC
    # ---------------------------------------------------------
    
    # 1. Sort Rows (Events) by Total Attendance
    # We want Highest at the TOP.
    # standard plot(kind='barh') plots index 0 at bottom.
    # So we sort Ascending (Smallest -> Largest), so Largest is at bottom of DF, 
    # then it plots at top? No, usually index 0 is bottom.
    # Let's sort Ascending (Smallest first). 
    # Then df_pivot.iloc[-1] is the Largest.
    # If we plot, the last row usually goes to the top.
    df_pivot['total'] = df_pivot.sum(axis=1)
    df_pivot = df_pivot.sort_values(by='total', ascending=True) # Smallest at top of DF (index 0), Largest at bottom
    df_pivot = df_pivot.drop(columns=['total'])

    # 2. Sort Columns (Universities) by Size (Largest on Left)
    # We calculate total for each uni in this specific slice
    uni_totals = df_pivot.sum(axis=0).sort_values(ascending=False)
    sorted_unis = uni_totals.index.tolist()
    df_pivot = df_pivot[sorted_unis]

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    plt.figure(figsize=(16, max(6, len(df_pivot) * 0.6)))
    
    colors = [university_colors.get(col, '#333333') for col in df_pivot.columns]
    
    ax = df_pivot.plot(kind='barh', stacked=True, color=colors, width=0.7, ax=plt.gca())
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Total Attendance', fontsize=12)
    plt.ylabel('Workshop Title', fontsize=12)
    
    # Add numbers
    max_val = df_pivot.sum(axis=1).max()
    plt.xlim(0, max_val * 1.25)
    
    for i, event in enumerate(df_pivot.index):
        cumulative = 0
        for col in df_pivot.columns:
            val = df_pivot.loc[event, col]
            if val > 0:
                # Label in center of bar segment
                ax.text(cumulative + val/2, i, f'{int(val)}', 
                        ha='center', va='center', color='white', fontsize=9, fontweight='bold')
                cumulative += val
        
        # Total label at end
        ax.text(cumulative + max_val*0.01, i, f'Total: {int(cumulative)}', 
                ha='left', va='center', color='black', fontweight='bold')

    plt.legend(title='University Program', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# -----------------------------------------
# 6️⃣ Generate Graphs by Category & Sub-Category
# -----------------------------------------

# Get unique Categories
categories = df_attended['Category'].unique()

for cat in categories:
    print(f"\nProcessing Category: {cat}")
    
    # Get Sub-Categories for this Category
    sub_cats = df_attended[df_attended['Category'] == cat]['Sub-Category'].unique()
    
    for sub in sub_cats:
        print(f"  - Generating graph for Sub-Category: {sub}")
        
        # Filter data
        df_sub = df_attended[(df_attended['Category'] == cat) & (df_attended['Sub-Category'] == sub)]
        
        # Aggregate by Event
        df_agg = df_sub.groupby(['event_name', 'uni_category'])['is_attended'].sum().reset_index()
        
        # Calculate totals to find Top 10
        event_totals = df_agg.groupby('event_name')['is_attended'].sum().reset_index()
        top_10_events = event_totals.sort_values('is_attended', ascending=False).head(10)['event_name'].tolist()
        
        # Filter for Top 10
        df_plot = df_agg[df_agg['event_name'].isin(top_10_events)]
        
        plot_title = f'Top 10 Workshops: {cat} - {sub}'
        plot_stacked_bar(df_plot, plot_title, f"{cat}_{sub}")

print("\n✅ Analysis Complete.")
