import re
import pandas as pd

def set_age_data(df):
    # Age groups calculation
    age_bins = [0, 19, 40, 100]
    age_labels = ['12-19', '20-40', '40+']
    age_groups = pd.cut(df['patient_age'], bins=age_bins, labels=age_labels)
    age_dist = age_groups.value_counts(normalize=True).sort_index() * 100
    
    return age_dist

def set_sleep_data(df):
    # Sleep disorder data (including chronic fatigue)
    has_sleep_disorder = df['sleep_disorder_type'].notna() & (df['sleep_disorder_type'] != "")
    sleep_disorder_percentage = (has_sleep_disorder).mean() * 100
    
    return sleep_disorder_percentage

def set_tenderness_data(df):
    # Tenderness/Stiffness/Soreness data
    tenderness_metrics = ["muscle_tenderness_present", "muscle_stiffness_present", "muscle_soreness_present"]
    tenderness_percentage = (df[tenderness_metrics] == "True").any(axis=1).mean() * 100
    
    return tenderness_percentage

def set_migraine_data(df):
    no_migraine_headache = 0
    headache_only = 0
    migraine_only = 0
    migraine_and_headache = 0

    # Iterate through the dataframe to categorize patients
    for idx, row in df.iterrows():
        migraine_history = row['migraine_history']
        headache_intensity = row['headache_intensity']
        
        if pd.isna(migraine_history) or migraine_history == "":
            if pd.isna(headache_intensity) or headache_intensity == 0:
                no_migraine_headache += 1
            else:
                headache_only += 1
        else:
            if pd.isna(headache_intensity) or headache_intensity == 0:
                migraine_only += 1
            else:
                migraine_and_headache += 1

    # Calculate percentages
    total_patients = len(df)
    no_migraine_headache_pct = (no_migraine_headache / total_patients) * 100
    headache_only_pct = (headache_only / total_patients) * 100
    migraine_only_pct = (migraine_only / total_patients) * 100
    migraine_and_headache_pct = (migraine_and_headache / total_patients) * 100
    
    return no_migraine_headache_pct, headache_only_pct, migraine_only_pct, migraine_and_headache_pct

def set_left_stick_data(df):
    metrics = ["headache_intensity", "average_daily_pain_intensity", 
              "diet_score", "tmj_pain_rating", "disability_rating"]
    
    metrics_titles = ["Headache\nIntensity", "Daily Pain\nIntensity", "Diet\nScore", "TMJ pain\nRating", "Disability\nRating"]
    
    means = df[metrics].mean()
    means = means.fillna(0) if means.isna().any() else means
    std_devs = df[metrics].std()
    std_devs = std_devs.fillna(0) if std_devs.isna().any() else std_devs
    
    return metrics_titles, means, std_devs

def set_middle_stick_data(df):
    def extract_mm_value(value):
        """
        Extract numerical value from a string like '33mm'.
        Returns NaN if no match is found.
        """
        match = re.search(r'\d+', str(value))
        return float(match.group(0)) if match else 0

    # Apply the function to extract numerical values
    max_opening = df['maximum_opening'].apply(extract_mm_value) if 'maximum_opening' in df else pd.Series([0] * len(df))
    max_opening_no_pain = df['maximum_opening_without_pain'].apply(extract_mm_value) if 'maximum_opening_without_pain' in df else pd.Series([0] * len(df))
    
    return max_opening, max_opening_no_pain

def set_right_stick_data(df):
    possible_locations = {
        "frontal": ["frontal", "forehead"],
        "temporal": ["temporal", "side of head"],
        "posterior": ["posterior", "back of head"],
        "top of the head": ["top of the head", "vertex"],
        "temple": ["temple"]
    }

    location_counts = {key: 0 for key in possible_locations}

    for locations in df['headache_location']:
        if pd.notna(locations) and locations != "":
            locations_lower = locations.lower()
            for key, synonyms in possible_locations.items():
                if any(re.search(rf"\b{re.escape(syn)}\b", locations_lower) for syn in synonyms):
                    location_counts[key] += 1

    total_patients = len(df)
    location_percentages = {location: (count / total_patients) * 100 
                            for location, count in location_counts.items()}
    
    return location_percentages


def set_joint_pain_data(df):
    possible_areas = ["TMJ", "Neck", "Shoulder", "Back"]
    joint_pain_counts = {area: 0 for area in possible_areas}

    for areas in df['joint_pain_areas']:
        if pd.notna(areas) and areas != "":
            areas_lower = areas.lower()
            # Match "left TMJ", "right TMJ", or phrases like "TMJ pain on left side"
            if re.search(r'\bleft\b.*tmj|\btmj.*left|\bright\b.*tmj|\btmj.*right|\btmj\b', areas_lower):
                joint_pain_counts["TMJ"] += 1
            for area in possible_areas:
                if area != "TMJ" and area.lower() in areas_lower:
                    joint_pain_counts[area] += 1

    total_patients = len(df)
    joint_pain_percentages = {area: (count / total_patients) * 100 
                              for area, count in joint_pain_counts.items()}
    
    return joint_pain_percentages

def set_upper_donuts_data(df):
    bool_metrics = ["earache_present", "tinnitus_present", "vertigo_present", 
                "hearing_loss_present", "jaw_crepitus", "jaw_clicking"]

    # Calculate percentages for boolean metrics
    bool_percentages = {metric: (df[metric] == "True").mean() * 100 
                        for metric in bool_metrics}

    # Calculate percentage for jaw issues (crepitus or clicking)
    jaw_issues = (df['jaw_crepitus'] != "") | (df['jaw_clicking'] != "")
    jaw_issues_percentage = jaw_issues.mean() * 100

    # Calculate percentage for mental health issues (anxiety, depression, or stress)
    mental_health_issues = (df['anxiety_present'] == "True") | \
                            (df['depression_present'] == "True") | \
                            (df['stress_present'] == "True")
    mental_health_percentage = mental_health_issues.mean() * 100
    
    return bool_percentages, jaw_issues_percentage, mental_health_percentage

def extract_months_from_pain_onset(pain_onset_str):
    """
    Convert pain_onset_date strings (e.g., "3 years ago", "1 year and 9 months ago") into total months.
    """
    if pd.isna(pain_onset_str) or pain_onset_str == "":
        return 0
    
    # Extract years and months using regex
    years_match = re.search(r'(\d+)\s*year', pain_onset_str)
    months_match = re.search(r'(\d+)\s*month', pain_onset_str)
    
    years = int(years_match.group(1)) if years_match else 0
    months = int(months_match.group(1)) if months_match else 0
    
    return years * 12 + months

def set_pain_onset_data(df):
    # Define age bins (10-year intervals)
    age_bins = list(range(10, 101, 10))  # [10, 20, 30, ..., 100]
    age_labels = [f'{start}-{start+9}' for start in age_bins[:-1]]  # ['10-19', '20-29', ..., '90-99']

    # Convert pain_onset_date to months
    df['pain_onset_months'] = df['pain_onset_date'].apply(extract_months_from_pain_onset)

    # Add a column for age bins
    df['age_bin'] = pd.cut(df['patient_age'], bins=age_bins, labels=age_labels, right=False)

    # Group by age bin and calculate mean and SD of patient ages
    age_distribution_stats = df.groupby('age_bin', observed=False)['patient_age'].agg(['mean', 'std']).reset_index()

    # Group by age bin and calculate mean and SD of pain onset duration (in years)
    pain_onset_stats = df.groupby('age_bin', observed=False)['pain_onset_months'].agg(['mean', 'std']).reset_index()
    pain_onset_stats = pain_onset_stats.dropna(subset=['mean', 'std'])
    pain_onset_stats['mean'] /= 12  # Convert mean to years
    pain_onset_stats['std'] /= 12  # Convert SD to years

    # Merge the two datasets
    merged_stats = pd.merge(age_distribution_stats, pain_onset_stats, on='age_bin', suffixes=('_age', '_pain'))

    # Filter out age bins with no data
    merged_stats = merged_stats.dropna(subset=['mean_age', 'mean_pain'])

    # Data for plotting
    age_bins_plot = merged_stats['age_bin']
    mean_ages = merged_stats['mean_age']
    std_ages = merged_stats['std_age']
    mean_pain = merged_stats['mean_pain']
    std_pain = merged_stats['std_pain']
    
    return age_bins_plot, mean_ages, std_ages, mean_pain, std_pain

def set_disc_displacement_data(df):
    """
    Calculate disc displacement percentages for left and right TMJ.
    
    Args:
        df (pd.DataFrame): Processed dataframe from process_summaries()
    
    Returns:
        tuple: (left_tmj_data, right_tmj_data) where each is a dictionary of percentages
    """
    # Initialize counts
    left_counts = {
        "w/o reduction": 0,
        "w/ reduction": 0,
        "reduction not specified": 0,
        "no displacement": 0
    }
    
    right_counts = {
        "w/o reduction": 0,
        "w/ reduction": 0,
        "reduction not specified": 0,
        "no displacement": 0
    }
    
    # Iterate through the dataframe
    for _, row in df.iterrows():
        disc_displacement = str(row['disc_displacement']).lower()  # Normalize casing
        
        # LEFT SIDE
        if re.search(r'left.*without reduction', disc_displacement):
            left_counts["w/o reduction"] += 1
        elif re.search(r'left.*with reduction', disc_displacement):
            left_counts["w/ reduction"] += 1
        elif 'left' in disc_displacement:
            left_counts["reduction not specified"] += 1
        else:
            left_counts["no displacement"] += 1

        # RIGHT SIDE
        if re.search(r'right.*without reduction', disc_displacement):
            right_counts["w/o reduction"] += 1
        elif re.search(r'right.*with reduction', disc_displacement):
            right_counts["w/ reduction"] += 1
        elif 'right' in disc_displacement:
            right_counts["reduction not specified"] += 1
        else:
            right_counts["no displacement"] += 1
    
    # Convert counts to percentages
    total_patients = len(df)
    left_tmj_data = {k: (v / total_patients) * 100 for k, v in left_counts.items()}
    right_tmj_data = {k: (v / total_patients) * 100 for k, v in right_counts.items()}
    
    return left_tmj_data, right_tmj_data


def set_muscle_pain_data(df):
    pain_levels = ["mild", "mild to moderate", "moderate", "moderate to severe", "severe"]
    pain_counts = {level: 0 for level in pain_levels}

    for score in df['muscle_pain_score']:
        if pd.isna(score) or score.strip() == "":
            continue
        score = score.lower()

        # Normalize punctuation and remove hyphens
        score = score.replace("-", " ").strip()

        matched = False
        # Match any mild-to-moderate variation
        if re.search(r"\bmild\s+to\s+moderate\b", score) or re.search(r"\bmoderate\s+to\s+mild\b", score):
            pain_counts["mild to moderate"] += 1
            matched = True
        # Match moderate-to-severe or similar phrases
        elif re.search(r"\bmoderate\s+to\s+severe\b", score) or "high moderate" in score or "high moderate to low severe" in score:
            pain_counts["moderate to severe"] += 1
            matched = True
        elif re.search(r"\bmild\w*\b", score) or re.search(r"\blow\b", score):
            pain_counts["mild"] += 1
            matched = True
        elif re.search(r"\bmoderate\b", score) and not re.search(r"moderate\s+to\s+severe", score):
            pain_counts["moderate"] += 1
            matched = True
        elif re.search(r"\bsevere\b", score) and not re.search(r"moderate\s+to\s+severe", score):
            pain_counts["severe"] += 1
            matched = True

    total_patients = len(df)
    pain_percentages = {level: (pain_counts[level] / total_patients) * 100 for level in pain_levels}

    clean_pain_levels = [
        level.replace("mild to moderate", "mild\nto\nmoderate").replace("moderate to severe", "moderate\nto\nsevere")
        for level in pain_levels
    ]

    return clean_pain_levels, pain_percentages


def initialize_key_value_summary():
    """
    Initialize a dictionary with default values based on the expected types.

    Returns:
        dict: Dictionary with default values assigned based on the types.
    """
    KEYS_AND_TYPES = {
    "patient_id": str,
    "patient_age": str,
    "headache_intensity": str,
    "headache_frequency": str,
    "headache_location": str,
    "migraine_history": str,
    "migraine_frequency": str,
    "average_daily_pain_intensity": str,
    "diet_score": str,
    "tmj_pain_rating": str,
    "disability_rating": str,
    "jaw_function_score": str,
    "jaw_clicking": str,
    "jaw_crepitus": str,
    "jaw_locking": str,
    "maximum_opening": str,
    "maximum_opening_without_pain": str,
    "disc_displacement": str,
    "muscle_pain_score": str,
    "muscle_pain_location": str,
    "muscle_spasm_present": str,
    "muscle_tenderness_present": str,
    "muscle_stiffness_present": str,
    "muscle_soreness_present": str,
    "joint_pain_areas": str,
    "joint_arthritis_location": str,
    "neck_pain_present": str,
    "back_pain_present": str,
    "earache_present": str,
    "tinnitus_present": str,
    "vertigo_present": str,
    "hearing_loss_present": str,
    "hearing_sensitivity_present": str,
    "sleep_apnea_diagnosed": str,
    "sleep_disorder_type": str,
    "airway_obstruction_present": str,
    "anxiety_present": str,
    "depression_present": str,
    "stress_present": str,
    "autoimmune_condition": str,
    "fibromyalgia_present": str,
    "current_medications": str,
    "previous_medications": str,
    "adverse_reactions": str,
    "appliance_history": str,
    "current_appliance": str,
    "cpap_used": str,
    "apap_used": str,
    "bipap_used": str,
    "physical_therapy_status": str,
    "pain_onset_date": str,
    "pain_duration": str,
    "pain_frequency": str,
    "onset_triggers": str,
    "pain_relieving_factors": str,
    "pain_aggravating_factors": str
    }
    
    defaults = {
        str: "",
    }
    return {key: defaults[expected_type] for key, expected_type in KEYS_AND_TYPES.items()}