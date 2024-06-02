import xml.etree.ElementTree as ET
import os

# Input and output file paths
input_file = "ECB/ecb-en.xml"
output_file = "ECB/extracted/ecb_en.txt"

os.makedirs(os.path.join("ECB/extracted", output_file), exist_ok=True)

# Parse the XML file
tree = ET.parse(input_file)
root = tree.getroot()

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as f_out:
    # Iterate over each sentence element
    for s in root.findall('.//s'):
        # Extract all word elements within the sentence
        words = [w.text for w in s.findall('w')]
        # Join the words to form the complete sentence
        sentence = ' '.join(words)
        # Write the sentence to the output file
        f_out.write(sentence + '\n')

print(f"Sentences extracted and saved to {output_file}")
