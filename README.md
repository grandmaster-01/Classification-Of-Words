# Classification of Words

## Project Overview
This repository contains **Project 5: Word Classification and Parent Word Identification** - a C programming project that implements a word processing system to classify words by length and identify parent-child word relationships.

## What's in the Zip File?
The `Classification Of Words.zip` file contains:

### Code Files
- **`word_processor.h`** - Header file with structure definitions and function prototypes
- **`word_processor_main.c`** - Main implementation with all core functionality
- **`word_processor_main.exe`** - Compiled Windows executable

### Source Files
- **`project_text.txt`** - Input text file containing "The Adventure of Silver Blaze" by Arthur Conan Doyle
- **`project5_data.csv`** - CSV file with subset words to search for

### Documentation
- **`PROJECT5.pdf`** - Project specification and requirements document (13 pages)

## Project Features

### 1. Word Classification by Length
- Classifies words into linked lists based on their length
- Supports words with lengths from 4 to 12 characters
- Uses dynamic data structures (linked lists) for efficient storage

### 2. Subset-Parent Word Relationship
- Identifies "parent" words that contain all characters of a "subset" word in sequential order
- Example: "vent" is a subset of "adventure" because all letters of "vent" appear in "adventure" in order
- Removes duplicate parent words from results

### 3. Data Processing
- Reads words from text files
- Parses CSV files for subset word lists
- Filters out non-alphabetic words
- Handles words of varying lengths efficiently

## Technical Implementation

### Data Structures
- **Linked List**: Used to organize words by length
- **Arrays**: Store words read from files

### Key Functions
- `createNode()` - Creates new linked list nodes
- `isAlphabetic()` - Validates alphabetic-only words
- `insertWord()` - Inserts words into appropriate linked lists
- `containsAllCharacters()` - Checks sequential character matching
- `readWordsFromFile()` - Reads words from text files
- `readSubsetWordsFromCSV()` - Parses CSV files
- `removeDuplicateParents()` - Eliminates duplicate results

### Algorithm
The program uses a sequential character matching algorithm where:
1. Words are read from the input text file
2. Words are classified by length into linked lists
3. Subset words are loaded from CSV
4. For each subset word, all words are checked for sequential character containment
5. Parent words are displayed with their lengths
6. Duplicate parents are removed

## How to Use

### Prerequisites
- C compiler (GCC, MinGW, or similar)
- Text editor or IDE

### Setup
First, extract the `Classification Of Words.zip` file to access the project files:
```bash
unzip "Classification Of Words.zip"
cd Project5
```

### Compilation
Navigate to the Code Files directory and compile:
```bash
cd "Code Files"
gcc word_processor_main.c -o word_processor_main
```

### Execution
The program expects input files in the current directory. Copy them from Source Files or run from the Project5 root:
```bash
# Option 1: Copy files to Code Files directory
cp "../Source Files/project_text.txt" .
cp "../Source Files/project5_data.csv" .
./word_processor_main

# Option 2: Copy files to Project5 root and run from there
cd ..
cp "Source Files/project_text.txt" .
cp "Source Files/project5_data.csv" .
"./Code Files/word_processor_main"
```
Note: You may need to modify the file paths in the code if running from different directories.

### Expected Output
The program displays:
- Each subset word with its length
- All parent words containing that subset
- Length of each parent word
- Total count of parent words for each subset

## Example
```
Subset Words and their Parent Words:

Subset: vent (length: 4)
  - Parent: adventure (length: 9)
  - Parent: prevent (length: 7)
  - Parent: event (length: 5)
Total Parent Words: 3
```

## Configuration
Key constants are defined in `word_processor.h` (found inside the zip file):
- `MAX_WORD_LENGTH`: 12 characters
- `MIN_WORD_LENGTH`: 4 characters
- `MAX_WORDS`: 1000 words

To modify these settings, extract the zip file, edit `word_processor.h`, and recompile.

## Technologies
- **Language**: C
- **Data Structures**: Linked Lists, Arrays
- **File I/O**: Text files (.txt) and CSV files (.csv)
- **Memory Management**: Dynamic allocation with malloc/free

## Project Type
Data Structures and Algorithms (DSA) course project focusing on linked lists, file I/O, and string processing algorithms.