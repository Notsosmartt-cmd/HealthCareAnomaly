import os
import glob
import sys


target_directory = "../Data_Sets/Pima"

# Correct attribute names in order (first 8 attributes)
NEW_ATTRIBUTES = [
    'pregnancies',
    'glucose',
    'bloodPressure',
    'SkinThickness',
    'Insulin',
    'bmi',
    'diabetesPedigree',
    'age'
]


def process_arff_files(directory):
    """Process all ARFF files in the specified directory"""
    arff_files = glob.glob(os.path.join(directory, '*.arff'))

    if not arff_files:
        print(f"\nNo ARFF files found in {directory}")
        return

    print(f"\nProcessing directory: {directory}")
    print(f"Found {len(arff_files)} ARFF file(s) to process")

    for filepath in arff_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")

        with open(filepath, 'r') as f:
            lines = f.readlines()

        new_lines = []
        attr_index = 0
        in_data_section = False
        needs_processing = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('%'):
                new_lines.append(line)
                continue

            lower_line = stripped.lower()

            if in_data_section:
                new_lines.append(line)
            elif lower_line.startswith('@relation'):
                new_lines.append(line)
            elif lower_line.startswith('@attribute'):
                parts = [p.strip() for p in line.split("'", 2)]

                # Check if first attribute is generic (att1)
                if attr_index == 0 and len(parts) > 2 and parts[1].lower() == 'att1':
                    needs_processing = True
                    print("  Found generic attributes. Starting replacement...")

                # Replace attribute name if processing needed
                if needs_processing and attr_index < len(NEW_ATTRIBUTES):
                    new_name = NEW_ATTRIBUTES[attr_index]
                    new_line = f"@attribute '{new_name}' {parts[2]}\n"
                    new_lines.append(new_line)
                    print(f"  Replaced '{parts[1]}' → '{new_name}'")
                    attr_index += 1
                else:
                    new_lines.append(line)
                    attr_index += 1

            elif lower_line.startswith('@data'):
                in_data_section = True
                new_lines.append(line)
            else:
                new_lines.append(line)

        # Save changes if processing occurred
        if needs_processing:
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            print("  File updated successfully")
        else:
            print("  No changes needed (no generic attributes found)")


if __name__ == "__main__":
    # Verify target directory exists
    if not os.path.isdir(target_directory):
        print(f"\nERROR: Directory not found - {target_directory}")
        print("Please update the 'target_directory' variable in the script")
        print("Creating directory structure is not supported. Please create the directory first.")
        sys.exit(1)

    process_arff_files(target_directory)
    print("\nProcessing complete!")