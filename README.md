# Classification Of Words

A C program that classifies and analyzes word relationships by finding "parent words" that contain subset words in sequential character order.

## Overview

This project implements a word processing system that:
- Reads words from a text file and organizes them into linked lists based on word length
- Processes subset words from a CSV file
- Identifies parent words that contain all characters of a subset word in sequential order
- Removes duplicate parent word matches
- Displays the relationship between subset words and their parent words

## Features

- **Linked List Organization**: Words are stored in linked lists categorized by their length (4-12 characters)
- **Sequential Character Matching**: Finds parent words where subset characters appear in order (not necessarily consecutive)
- **Duplicate Removal**: Automatically removes duplicate parent word matches
- **File Input Support**: Reads from text files and CSV files
- **Memory Management**: Proper memory allocation and deallocation for linked lists

## Project Structure

```
Classification Of Words/
├── README.md
└── Classification Of Words.zip
    └── Project5/
        ├── Code Files/
        │   ├── word_processor.h          # Header file with function prototypes
        │   ├── word_processor_main.c     # Main program implementation
        │   └── word_processor_main.exe   # Compiled executable
        ├── Source Files/
        │   ├── project5_data.csv         # CSV file with subset words
        │   └── project_text.txt          # Text file with words to analyze
        └── PROJECT5.pdf                  # Project documentation
```

## How to Compile

```bash
# Navigate to the Code Files directory
cd "Project5/Code Files"

# Compile the program using GCC
gcc -o word_processor word_processor_main.c -std=c99

# Or use the pre-compiled executable
./word_processor_main.exe
```

## How to Run

1. Extract the `Classification Of Words.zip` file
2. Navigate to `Project5/Code Files/`
3. Ensure the input files are in the same directory or provide correct paths:
   - `project_text.txt` - Main word source file
   - `project5_data.csv` - Subset words to search for
4. Run the executable:
   ```bash
   ./word_processor_main.exe
   ```

## Input File Formats

### Text File (project_text.txt)
- Plain text file containing words separated by spaces or newlines
- Example: "The Adventure of Silver Blaze..."
- Program extracts and processes individual words

### CSV File (project5_data.csv)
- First row is the header (ignored)
- Each subsequent row contains one subset word
- Example:
  ```
  Subset Words
  vent
  lack
  read
  ```

## Example Usage

Given the subset word "vent" and parent words like "adventure", "event", "prevent", the program will:
1. Check if 'v', 'e', 'n', 't' appear in sequence in each word
2. Identify matches: "adventure" contains v-e-n-t in sequence (ad-**v**-**e**-**n**-**t**-ure)
3. Display all parent words with their lengths
4. Show the total count of unique parent words

### Sample Output
```
Subset Words and their Parent Words:

Subset: vent (length: 4)
  - Parent: adventure (length: 9)
  - Parent: event (length: 5)
Total Parent Words: 2
```

## Technical Details

- **Language**: C (C99 standard)
- **Word Length Range**: 4-12 characters
- **Maximum Words**: Up to 1000 words from the text file and up to 1000 subset words from the CSV file
- **Data Structure**: Array of linked lists for efficient word organization
- **Character Validation**: Only alphabetic characters are processed

## Functions

- `createNode()` - Creates a new linked list node
- `isAlphabetic()` - Validates alphabetic characters
- `insertWord()` - Inserts words into appropriate linked lists
- `freeLists()` - Deallocates memory for linked lists
- `containsAllCharacters()` - Checks sequential character matching
- `readWordsFromFile()` - Reads words from text file
- `readSubsetWordsFromCSV()` - Reads subset words from CSV
- `removeDuplicateParents()` - Removes duplicate parent word matches

## Memory Management

The program properly manages memory by:
- Allocating nodes dynamically using `malloc()`
- Freeing all allocated memory using `freeLists()` before program termination
- Ensuring null-termination of all strings

## License

This is an educational project for learning data structures and algorithms.