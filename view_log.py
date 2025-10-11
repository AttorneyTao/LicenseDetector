# Read the log file
with open('logs/llm_interaction.log', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find copyright analysis entries
entries = []
in_copyright_entry = False
current_entry = []

for line in lines:
    if 'Copyright Notice Construction' in line:
        if not in_copyright_entry:
            # Start a new entry
            in_copyright_entry = True
            current_entry = [line]
        else:
            # Add to current entry
            current_entry.append(line)
    elif in_copyright_entry:
        if line.startswith('2025-'):
            # End of entry, save it
            entries.append(''.join(current_entry))
            in_copyright_entry = False
        else:
            # Continue current entry
            current_entry.append(line)

# Add the last entry if needed
if in_copyright_entry:
    entries.append(''.join(current_entry))

# Print copyright analysis entries
for i, entry in enumerate(entries):
    print(f"Entry {i+1}:")
    print(entry)
    print('-' * 80)