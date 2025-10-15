import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from MedX_CLI_utils.dashboard_utils import (
    set_left_stick_data, set_middle_stick_data, set_right_stick_data,
    set_upper_donuts_data, set_age_data, set_sleep_data, set_tenderness_data,
    set_migraine_data, set_muscle_pain_data, set_pain_onset_data,
    set_disc_displacement_data, set_joint_pain_data, initialize_key_value_summary
)

def read_summaries(summary_folder: str) -> dict:
    """
    Read patient summary files from a directory and return as a dictionary.
    
    Args:
        summary_folder (str): Path to folder containing summary text files
    
    Returns:
        dict: Dictionary with patient IDs as keys and summary contents as values
    """
    summaries = {}
    for file_name in os.listdir(summary_folder):
        if file_name.lower().endswith("_pred.txt"):
            patient_id = file_name.split("_")[0]
            with open(os.path.join(summary_folder, file_name), "r", encoding="utf-8") as file:
                summaries[patient_id] = file.read()
    return summaries

def extract_numbers(value: str) -> float:
    """
    Extract numbers from a string and return their mean.
    
    Args:
        value (str): Input string containing numerical values
    
    Returns:
        float: Mean of extracted numbers, or np.nan if no numbers found
    """
    # If value contains '/', only take the number before the first '/'
    if '/' in value:
        first_part = value.split('/')[0]
        match = re.search(r'\d+(?:\.\d+)?', first_part)
        return float(match.group(0)) if match else np.nan
    numbers = [float(num) for num in re.findall(r'\d+(?:\.\d+)?', value)]
    return np.mean(numbers) if numbers else np.nan

def update_dictionary(patient_dict: dict, chunk: str) -> None:
    """
    Update patient dictionary with values from a summary chunk using smart merging.
    
    Args:
        patient_dict (dict): Dictionary to update with patient data
        chunk (str): Section of summary text containing key-value pairs
    """
    for line in chunk.split("\n"):
        key_value = line.split("=")
        if len(key_value) != 2:
            continue
        key, value = key_value[0].strip(), key_value[1].strip()
        # Remplace les ';' par ',' dans la valeur
        value = value.replace(';', ',')
        
        # Skip patient_id to prevent overwriting/merging
        if key == "patient_id":
            continue
            
        if key not in patient_dict:
            continue
        
        current_value = patient_dict[key]
        
        # Handle boolean fields
        if current_value in {"True", "False"}:
            if current_value == "False" and value == "True":
                patient_dict[key] = "True"
            elif current_value == "True" and value == "False":
                patient_dict[key] = "False"
            continue
        
        # Handle empty values
        if current_value == "":
            patient_dict[key] = value
            continue
            
        # Smart merge for string values
        current_parts = current_value.split(" | ")
        
        # Skip if value already exists
        if value in current_parts:
            continue
            
        # Replace existing substrings with more detailed values
        new_parts = []
        replaced = False
        for part in current_parts:
            if part in value and part != value:
                new_parts.append(value)
                replaced = True
            else:
                new_parts.append(part)
        
        # Deduplicate if replacements occurred
        if replaced:
            new_parts = list(dict.fromkeys(new_parts))  # preserve order
            if value in new_parts:
                patient_dict[key] = " | ".join(new_parts)
                continue
        
        # Check if new value is redundant
        redundant = any(value in part for part in new_parts)
        
        # Add new value if not redundant
        if not redundant:
            new_parts.append(value)
        
        patient_dict[key] = " | ".join(new_parts)

def process_summaries(summaries: dict) -> pd.DataFrame:
    """
    Process raw summary data into a structured DataFrame.
    
    Args:
        summaries (dict): Dictionary of patient summaries from read_summaries()
    
    Returns:
        pd.DataFrame: Processed dataframe with numerical conversions
    """
    patient_data = []
    
    for patient_id, summary in summaries.items():
        patient_dict = initialize_key_value_summary()
        patient_dict["patient_id"] = patient_id
        update_dictionary(patient_dict, summary)
        patient_data.append(patient_dict)
    
    df = pd.DataFrame(patient_data)

    # Convert numeric fields by extracting numbers and computing means
    numeric_fields = ["patient_age", "headache_intensity", "average_daily_pain_intensity", 
                      "diet_score", "tmj_pain_rating", "disability_rating"]

    for field in numeric_fields:
        df[field] = df[field].apply(lambda x: extract_numbers(str(x)))

    return df

def generate_dashboard_figure(df: pd.DataFrame, output_folder: str = None) -> plt.Figure:
    """
    Generate and display a comprehensive patient data dashboard.
    
    Args:
        df (pd.DataFrame): Processed dataframe from process_summaries()
    """
    ## imports déjà présents en haut du fichier
    try:
        # output_folder est passé comme variable locale par show_dashboard
        import inspect
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        output_folder = values.get('output_folder', None)
    except Exception:
        output_folder = None
    if output_folder is None:
        output_folder = os.getcwd()
    csv_path = os.path.join(output_folder, "dashboard_full_dataframe.csv")
    df.to_csv(csv_path, index=False)

    # ================== Data Preparation ==================
    # Sticks data
    metrics_titles, means, std_devs = set_left_stick_data(df)
    max_opening, max_opening_no_pain = set_middle_stick_data(df)
    location_percentages = set_right_stick_data(df)
    
    # Upper donuts data
    bool_percentages, jaw_issues_percentage, mental_health_percentage = set_upper_donuts_data(df)
    
    # Upper left data
    age_dist = set_age_data(df)
    true_pct, unknown_pct = set_sleep_data(df)
    tenderness_percentage = set_tenderness_data(df)
    no_migraine_headache_pct, headache_only_pct, migraine_only_pct, migraine_and_headache_pct, unknown_pct = set_migraine_data(df)
    
    # Lower right data
    pain_levels, pain_percentages = set_muscle_pain_data(df)
    age_bins_plot, mean_ages, std_ages, mean_pain, std_pain = set_pain_onset_data(df)
    left_tmj_data, right_tmj_data = set_disc_displacement_data(df)


    # ================== Figure Setup ==================
    fig = plt.figure(figsize=(22, 14))
    plt.suptitle("Patient Data Dashboard", fontsize=24, fontweight='bold', y=0.97)
    fig.text(0.5, 0.93, f"Evaluation done using {len(df)} cases", fontsize=16, ha='center', va='center')
    
    # Main grid layout (2 rows, 2 columns)
    gs = fig.add_gridspec(2, 2, 
                         width_ratios=[3, 1.5], 
                         height_ratios=[1, 1.5],
                         hspace=0.2, wspace=0.1)





    # ================== Top Left: 4 Columns ==================
    gs_top_left = gs[0, 0].subgridspec(1, 4, wspace=0.2)

    # Column 1: Age Distribution
    ax_age = fig.add_subplot(gs_top_left[0, 0])
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    ax_age.pie(age_dist, labels=age_dist.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 10},
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
    ax_age.set_title('Age Distribution', fontsize=16, pad=15, fontweight='semibold')

    # Column 2: Sleep Disorder Prevalence (adapté deux segments)
    ax_sleep = fig.add_subplot(gs_top_left[0, 1])
    sleep_colors = ['#ff9999', '#bdbdbd']  # rouge, gris
    wedges, texts, autotexts = ax_sleep.pie(
        [true_pct, unknown_pct],
        labels=None,
        autopct='%1.1f%%',
        colors=sleep_colors,
        startangle=90,
        wedgeprops={'width': 1, 'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 10}
    )
    ax_sleep.set_title('Sleep Disorder Prevalence', fontsize=16, pad=15, fontweight='semibold')
    ax_sleep.set_aspect('equal')

    # Add a legend at the bottom
    legend_labels = ['With Sleep Disorder', 'Unknown']
    ax_sleep.legend(
        wedges,
        legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        fontsize=10,
        frameon=False
    )

    # Column 3: Tenderness/Stiffness/Soreness
    ax_tenderness = fig.add_subplot(gs_top_left[0, 2])
    true_pct, false_pct, unknown_pct = set_tenderness_data(df)
    tenderness_colors = ['#ffcc99', '#c2c2f0', '#bdbdbd']  # orange, violet, gris
    wedges, texts, autotexts = ax_tenderness.pie(
        [true_pct, false_pct, unknown_pct],
        labels=None,
        autopct='%1.1f%%',
        colors=tenderness_colors,
        startangle=90,
        wedgeprops={'width': 1, 'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 10}
    )
    ax_tenderness.set_title('Muscle Tenderness &\nStiffness & Soreness', fontsize=16, pad=-15, fontweight='semibold')
    ax_tenderness.set_aspect('equal')

    # Add a legend at the bottom
    legend_labels = ['With Symptoms', 'Without Symptoms', 'Unknown']
    ax_tenderness.legend(
        wedges,
        legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        fontsize=10,
        frameon=False
    )

    # Column 4: Migraine History
    ax_migraine = fig.add_subplot(gs_top_left[0, 3])  # Migraine subplot position

    # Create the pie chart with unknown segment
    migraine_data = [headache_only_pct, migraine_and_headache_pct, unknown_pct]
    migraine_labels = [ 'Headache', 'Migraine & Headache', 'Unknown']
    migraine_colors = [ '#fc8d62', '#e78ac3', '#bdbdbd']  # Add grey for unknown
    
    
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct >= 5 else ''

    wedges, texts, autotexts = ax_migraine.pie(
        migraine_data,
        labels=None,  # Remove default labels
        autopct=autopct_format,  # Custom autopct function
        colors=migraine_colors,
        startangle=90,
        wedgeprops={'width': 1, 'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 10}
    )

    # Add leader lines for small slices
    for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
        if migraine_data[i] < 5:  # Adjust threshold as needed
            angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            x = 1.2 * np.cos(np.deg2rad(angle))
            y = 1.2 * np.sin(np.deg2rad(angle))
            ax_migraine.annotate(
                f'{migraine_data[i]:.1f}%',
                xy=(np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="-", color='black'),
                fontsize=10,
                ha='center',
                va='center'
            )
            
    # Update the title
    ax_migraine.set_title('Migraine & Headache', fontsize=16, pad=15, fontweight='semibold')

    # Add a legend
    ax_migraine.legend(
        wedges,  # Use the pie chart wedges for color reference
        migraine_labels,  # Labels for the legend
        loc='lower center',  # Position the legend at the bottom
        bbox_to_anchor=(0.5, -0.15),  # Adjust vertical position
        ncol=2,  # Display legend items in 2 columns
        fontsize=10,  # Legend font size
        frameon=False  # Remove legend border
    )

    ax_migraine.set_aspect('equal')  # Ensure the pie chart is circular





    # ================== Bottom Left: 3 Columns ==================
    gs_bottom_left = gs[1, 0].subgridspec(1, 3, width_ratios=[5, 2, 1], wspace=0.2)

    # Column 1: Clinical Scores with SD
    ax_scores = fig.add_subplot(gs_bottom_left[0, 0])
    bars = ax_scores.bar(metrics_titles, means, yerr=std_devs,
                        color='#1f77b4', alpha=0.7, capsize=8, error_kw={'elinewidth': 2})
    
    # Formatting scores plot
    ax_scores.set_ylabel("Mean Score (0-10 Scale)", fontsize=14)
    ax_scores.set_ylim(0, 10)
    ax_scores.tick_params(axis='x', rotation=22.5, labelsize=10)
    ax_scores.grid(axis='y', linestyle='--', alpha=0.5)
    ax_scores.set_title("Clinical Scores with Standard Deviation", 
                       fontsize=16, pad=15, fontweight='semibold')
    
    # Add value labels with SD
    for bar, mean, sd in zip(bars, means, std_devs):
        yval = bar.get_height()
        ax_scores.text(bar.get_x() + bar.get_width()/2., 
                    yval + sd + 0.3,  # Add SD value + small offset
                    f'{mean:.1f} ± {sd:.1f}',
                    ha='center', 
                    va='bottom', 
                    fontsize=12,
                    color='#2d3436')

    # Column 2: Maximum Opening Data
    ax_max_opening = fig.add_subplot(gs_bottom_left[0, 1])
    max_opening_means = [max_opening.mean(), max_opening_no_pain.mean()]
    max_opening_stds = [max_opening.std(), max_opening_no_pain.std()]
    max_opening_titles = ["Max Opening", "Max Opening\nWithout Pain"]
    
    bars = ax_max_opening.bar(max_opening_titles, max_opening_means, yerr=max_opening_stds,
                             color='#2ca02c', alpha=0.7, capsize=8, error_kw={'elinewidth': 2})
    
    # Formatting max opening plot
    ax_max_opening.set_ylabel("Opening (mm)", fontsize=14)
    ax_max_opening.tick_params(axis='x', rotation=22.5, labelsize=10)
    ax_max_opening.grid(axis='y', linestyle='--', alpha=0.5)
    ax_max_opening.set_title("Maximum Opening", fontsize=16, pad=15, fontweight='semibold')
    
    # Add value labels with SD
    for bar, mean, sd in zip(bars, max_opening_means, max_opening_stds):
        yval = bar.get_height()
        ax_max_opening.text(bar.get_x() + bar.get_width()/2., 
                          yval + sd + 0.3,  # Add SD value + small offset
                          f'{mean:.1f} ± {sd:.1f}',
                          ha='center', 
                          va='bottom', 
                          fontsize=12,
                          color='#2d3436')

    # Column 3: Headache Locations - Horizontal Bar Chart
    ax_headache = fig.add_subplot(gs_bottom_left[0, 2])

    # Sort locations by percentage (ascending for better horizontal bar visualization)
    sorted_locations = sorted(location_percentages.items(), key=lambda x: x[1])
    # Modify long labels to be on two lines
    def format_label(label):
        if label == "top of the head":
            return "Top of\nthe head"
        return label.capitalize()

    locations = [format_label(loc[0]) for loc in sorted_locations]
    percentages = [loc[1] for loc in sorted_locations]

    # Define colors for the bars
    bar_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

    # Plot horizontal bars
    bars = ax_headache.barh(locations, percentages, color=bar_colors, alpha=0.7)

    # Calculate threshold (1/3 of max percentage)
    max_pct = max(percentages) * 1.2 if percentages else 1
    threshold = max_pct / 3

    # Add percentage labels with smart positioning
    for bar in bars:
        width = bar.get_width()
        if width == 0:  # Special case for 0%
            ax_headache.text(1,  # Just past 0
                            bar.get_y() + bar.get_height()/2,
                            f'{width:.1f}%',
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            fontweight='bold')
        elif width < threshold:  # Small bars - label after bar
            ax_headache.text(width + 0.2*width,  # Right of bar
                            bar.get_y() + bar.get_height()/2,
                            f'{width:.1f}%',
                            ha='left',
                            va='center',
                            fontsize=10,
                            color='black',
                            fontweight='bold')
        else:  # Large bars - label inside bar (white text)
            ax_headache.text(width / 2,  # Middle of bar
                            bar.get_y() + bar.get_height()/2,
                            f'{width:.1f}%',
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            fontweight='bold')

    # Formatting
    ax_headache.set_title('Headache Locations', fontsize=16, pad=15, fontweight='semibold')
    ax_headache.set_xlabel('Percentage', fontsize=12)
    ax_headache.set_xlim(0, max(percentages)*1.2 if max(percentages) > 0 else 10)
    ax_headache.grid(axis='x', linestyle='--', alpha=0.5)




    # ================== Top right: Boolean Metrics Donut Charts ==================
    gs_right = gs[0, 1].subgridspec(2, 3, hspace=0.05, wspace=0.1)

    positions = {
        0: (0, 0),  # earache_present
        1: (0, 1),  # tinnitus_present
        2: (0, 2),  # vertigo_present
        3: (1, 0),  # hearing_loss_present
        4: (1, 1),  # jaw_crepitus or jaw_clicking
        5: (1, 2)   # mental health (anxiety, depression, stress)
    }

    # Metrics to display in the donut charts
    donut_metrics = [
        "earache_present", "tinnitus_present", "vertigo_present",
        "hearing_loss_present", "jaw_issues", "mental_health"
    ]

    # Percentages for the donut charts
    donut_percentages = {
        "earache_present": bool_percentages["earache_present"],
        "tinnitus_present": bool_percentages["tinnitus_present"],
        "vertigo_present": bool_percentages["vertigo_present"],
        "hearing_loss_present": bool_percentages["hearing_loss_present"],
        "jaw_issues": jaw_issues_percentage,
        "mental_health": mental_health_percentage
    }

    # Titles for the donut charts
    donut_titles = {
        "earache_present": "Earache",
        "tinnitus_present": "Tinnitus",
        "vertigo_present": "Vertigo",
        "hearing_loss_present": "Hearing Loss",
        "jaw_issues": "Jaw Clicking/Crepitus",
        "mental_health": "Mental Health Issues"
    }

    for i, metric in enumerate(donut_metrics):
        ax_pie = fig.add_subplot(gs_right[positions[i]])
        true_pct = donut_percentages[metric]
        
        # Create enhanced donut chart
        ax_pie.pie([true_pct, 100 - true_pct], 
                wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 1},
                colors=['#ff6b6b', '#f0f0f0'])
        
        # Central text with percentage
        ax_pie.text(0, 0.0, f'{true_pct:.1f}%', 
                    ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='#2d3436')
        
        # Metric title below chart
        ax_pie.set_title(donut_titles[metric], 
                        fontsize=11, pad=1, loc='center', color='#2d3436')
        
        ax_pie.set_aspect('equal')
        ax_pie.set_xlim(-1.5, 1.5)
        
        
        
        
        
    # ================== Bottom Right: 4 Quadrants ==================
    gs_bottom_right = gs[1, 1].subgridspec(2, 2, height_ratios=[1,1.25] ,hspace=0.3, wspace=0.2)

    # ================== Top Left: Muscle Pain Score Distribution ==================
    ax_muscle_pain = fig.add_subplot(gs_bottom_right[0, 0])
    # Create a horizontal line
    ax_muscle_pain.axhline(0, color='black', linewidth=2, alpha=0.7)

    # Plot vertical lines for each pain level
    for idx, (level, percentage) in enumerate(pain_percentages.items()):
        x_position = idx  # Position along the horizontal line
        ax_muscle_pain.vlines(x_position, ymin=0, ymax=percentage, color='#ff6b6b', linewidth=4, alpha=0.7)
        
        # Add text labels above the vertical lines
        ax_muscle_pain.text(
            x_position,  # X position
            percentage + 2,  # Y position (slightly above the line)
            f'{percentage:.1f}%',  # Text to display
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            fontsize=10,  # Font size
            color='black',  # Text color
            fontweight='bold'  # Bold text
        )

    # Add pain level labels below the horizontal line
    for idx, level in enumerate(pain_levels):
        ax_muscle_pain.text(
            idx,  # X position
            -6,  # Y position (below the line)
            level.capitalize(),  # Text to display
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            fontsize=8,  # Font size
            color='black',  # Text color
            fontweight='bold'  # Bold text
        )

    # Formatting
    ax_muscle_pain.set_title('Muscle Pain Score Distribution', fontsize=14, pad=10, fontweight='semibold')
    ax_muscle_pain.set_xlim(-0.5, len(pain_levels) - 0.5)  # Set x-axis limits
    ax_muscle_pain.set_ylim(-10, max(pain_percentages.values()) + 10)  # Set y-axis limits
    ax_muscle_pain.set_xticks([])  # Remove x-axis ticks
    ax_muscle_pain.set_yticks([])  # Remove y-axis ticks
    ax_muscle_pain.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines

    # ================== Top Right: Joint Pain Areas ==================
    ax_joint_pain = fig.add_subplot(gs_bottom_right[0, 1])

    # Get joint pain data
    joint_pain_percentages = set_joint_pain_data(df)

    # Sort by percentage (descending) and get the top 4
    sorted_joint_pain = sorted(joint_pain_percentages.items(), key=lambda x: x[1], reverse=True)[:4]
    joint_pain_areas = [area for area, _ in sorted_joint_pain]
    joint_pain_percentages_values = [percentage for _, percentage in sorted_joint_pain]

    # Define colors for the bars
    joint_pain_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

    # Plot vertical bars
    bars = ax_joint_pain.bar(joint_pain_areas, joint_pain_percentages_values, color=joint_pain_colors, alpha=0.7)

    # Add percentage labels on top of the bars
    for bar, percentage in zip(bars, joint_pain_percentages_values):
        ax_joint_pain.text(
            bar.get_x() + bar.get_width() / 2,  # X position (middle of the bar)
            bar.get_height() + 1,  # Y position (slightly above the bar)
            f'{percentage:.1f}%',  # Text to display
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            fontsize=10,  # Font size
            color='black',  # Text color
            fontweight='bold'  # Bold text
        )

    # Formatting
    ax_joint_pain.set_title('Joint Pain Areas', fontsize=14, pad=10, fontweight='semibold')
    ax_joint_pain.set_ylim(0, 100)  # Set y-axis limit to 100%
    ax_joint_pain.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines
    ax_joint_pain.set_xticks(range(len(joint_pain_areas)))  # Set x-ticks
    ax_joint_pain.set_xticklabels(joint_pain_areas, rotation=0, ha='center', fontsize=10)  # Rotate x-labels for readability
    ax_joint_pain.set_ylabel('Percentage', fontsize=12)

    # ================== Bottom Left of Bottom Right: Pain Onset by Age Group ==================
    ax_age_distribution = fig.add_subplot(gs_bottom_right[1, 0])

    # Define colors for the rectangles
    rectangle_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffcc99', '#c2c2f0', '#ff9999', '#66b3ff']

    # Plot rectangles for age distribution
    for idx, (age_bin, mean_age, std_age) in enumerate(zip(age_bins_plot, mean_ages, std_ages)):
        # Rectangle properties
        rect_height = 2 * std_age  # Height of the rectangle is 2 * SD
        rect_bottom = mean_age - std_age  # Bottom of the rectangle is mean - SD
        rect = plt.Rectangle(
            (idx - 0.4, rect_bottom),  # X position (left edge of the rectangle)
            0.8,  # Width of the rectangle
            rect_height,  # Height of the rectangle
            color=rectangle_colors[idx % len(rectangle_colors)],  # Color of the rectangle
            alpha=0.7  # Transparency
        )
        ax_age_distribution.add_patch(rect)

        # Add a vertical line for pain onset SD (centered on the mean age)
        ax_age_distribution.vlines(
            idx,  # X position (center of the rectangle)
            mean_age - std_pain[idx],  # Y start position (mean age - SD)
            mean_age + std_pain[idx],  # Y end position (mean age + SD)
            colors='#2d3436',  # Color of the line
            linewidths=1.5  # Line width
        )
        
        # Add value labels with mean and SD for pain onset
        ax_age_distribution.text(
            idx,  # X position (center of the rectangle)
            mean_age - std_pain[idx] - 0.3,  # Y position
            f'{mean_pain[idx]:.1f} ± {std_pain[idx]:.1f}',  # Text to display
            ha='center',  # Horizontal alignment
            va='top',  # Vertical alignment
            fontsize=8,  # Font size
            color='#2d3436',  # Text color
            fontweight='bold'  # Bold text
        )

    # Formatting for the y-axis (age distribution)
    ax_age_distribution.set_title('Age Distribution with Pain Onset SD', fontsize=14, pad=10, fontweight='semibold')
    ax_age_distribution.set_ylabel('Age (Years)', fontsize=12)
    ax_age_distribution.set_xlabel('Age Group', fontsize=12)
    ax_age_distribution.grid(axis='y', linestyle='--', alpha=0.5)  # Add grid lines
    ax_age_distribution.set_ylim(0, max(mean_ages) + max(std_ages) + 15)  # Set y-axis limit dynamically

    # Set x-ticks to the age bins
    ax_age_distribution.set_xticks(range(len(age_bins_plot)))  # Set x-ticks
    ax_age_distribution.set_xticklabels(age_bins_plot, rotation=22.5, ha='center', fontsize=10)  # Rotate x-labels for readability
                
    # ================== Bottom Right: Disc Displacement Donuts ==================
    gs_disc_displacement = gs_bottom_right[1, 1].subgridspec(1, 2, wspace=0.01)
    # Left TMJ
    ax_left_tmj = fig.add_subplot(gs_disc_displacement[0, 0])
    left_labels = ["w/o reduction", "w/ reduction", "reduction not specified", "no displacement"]
    left_sizes = [left_tmj_data["w/o reduction"], left_tmj_data["w/ reduction"], 
                left_tmj_data["reduction not specified"], left_tmj_data["no displacement"]]
    left_colors = ['#ff6b6b', '#ffb366', '#66c2a5', '#8da0cb']

    wedges, texts, autotexts = ax_left_tmj.pie(
        left_sizes,
        labels=None,  # Remove default labels
        autopct='%1.1f%%',
        colors=left_colors,
        startangle=90,
        wedgeprops={'width': 1, 'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 9}
    )

    ax_left_tmj.set_title('Left TMJ\nDisc\nDisplacement', fontsize=12, pad=0, fontweight='semibold')
    ax_left_tmj.legend(
        wedges,  # Use the pie chart wedges for color reference
        left_labels,  # Labels for the legend
        loc='lower center',  # Position the legend at the bottom
        bbox_to_anchor=(1.125, -0.6),  # Adjust vertical position
        ncol=1,  # Display legend items in 2 columns
        fontsize=10,  # Legend font size
        frameon=False  # Remove legend border
    )

    ax_left_tmj.set_aspect('equal')

    # Right TMJ
    ax_right_tmj = fig.add_subplot(gs_disc_displacement[0, 1])
    right_labels = ["w/o reduction", "w/ reduction", "reduction not specified", "no displacement"]
    right_sizes = [right_tmj_data["w/o reduction"], right_tmj_data["w/ reduction"], 
                right_tmj_data["reduction not specified"], right_tmj_data["no displacement"]]
    right_colors = ['#ff6b6b', '#ffb366', '#66c2a5', '#8da0cb']

    wedges, texts, autotexts = ax_right_tmj.pie(
        right_sizes,
        labels=None,  # Remove default labels
        autopct='%1.1f%%',
        colors=right_colors,
        startangle=90,
        wedgeprops={'width': 1, 'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 9}
    )

    ax_right_tmj.set_title('Right TMJ\nDisc\nDisplacement', fontsize=12, pad=0, fontweight='semibold')

    ax_right_tmj.set_aspect('equal')
    
    # Final adjustments
    plt.subplots_adjust(top=0.85, bottom=0.07, left=0.04, right=0.98)
    # plt.show()
    return fig

def show_dashboard(summary_folder, output_folder):
    summaries = read_summaries(summary_folder)
    df = process_summaries(summaries)
    fig = generate_dashboard_figure(df, output_folder)
    
    output_path = os.path.join(output_folder, "dashboard.png")
    fig.savefig(output_path)
    df.to_csv(os.path.join(output_folder, "patient_data.csv"), index=False)
    print(f"Saved processed data to {os.path.join(output_folder, 'patient_data.csv')}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dashboard from patient summaries")
    parser.add_argument("summary_folder", type=str, help="Folder containing patient summaries")
    parser.add_argument("output_folder", type=str, help="Folder to save processed data as CSV and dashboard")
    args = parser.parse_args()

    show_dashboard(args.summary_folder, args.output_folder)
