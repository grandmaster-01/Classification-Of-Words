#include "word_processor.h"

// Function to create a new node
Node* createNode(const char *word) {
    Node *newNode = (Node *)malloc(sizeof(Node));
    if (newNode == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    strncpy(newNode->word, word, MAX_WORD_LENGTH);
    newNode->word[MAX_WORD_LENGTH] = '\0'; // Ensure null-termination
    newNode->next = NULL;
    return newNode;
}

// Function to check if a word contains only alphabetic characters
bool isAlphabetic(const char *word) {
    for (int i = 0; word[i] != '\0'; i++) {
        if (!((word[i] >= 'a' && word[i] <= 'z') || (word[i] >= 'A' && word[i] <= 'Z'))) {
            return false;
        }
    }
    return true;
}

// Function to insert a word into the appropriate linked list
void insertWord(Node **lists, const char *word) {
    if (!isAlphabetic(word)) {
        return; // Skip words with special characters
    }
    int length = strlen(word);
    if (length >= MIN_WORD_LENGTH && length <= MAX_WORD_LENGTH) {
        int index = length - MIN_WORD_LENGTH;
        Node *newNode = createNode(word);
        newNode->next = lists[index];
        lists[index] = newNode;
    }
}

// Function to free the linked lists
void freeLists(Node **lists) {
    for (int i = 0; i <= MAX_WORD_LENGTH - MIN_WORD_LENGTH; i++) {
        Node *current = lists[i];
        while (current) {
            Node *temp = current;
            current = current->next;
            free(temp);
        }
    }
}

// Function to check if all characters in "subset" exist in "parent" in sequential order
bool containsAllCharacters(const char *subset, const char *parent) {
    int subsetIndex = 0;
    int subsetLength = strlen(subset);
    int parentLength = strlen(parent);
    for (int i = 0; i < parentLength && subsetIndex < subsetLength; i++) {
        if (parent[i] == subset[subsetIndex]) {
            subsetIndex++; // Move to the next character in the subset
        }
    }
    return subsetIndex == subsetLength; // True if all characters in subset were matched sequentially in parent
}

// Function to read words from a file
// Helper to strip punctuation from a word
void stripPunctuation(char *word) {
    int len = strlen(word);
    if (len == 0) return;
    
    // Remove trailing punctuation
    while (len > 0 && !((word[len-1] >= 'a' && word[len-1] <= 'z') || (word[len-1] >= 'A' && word[len-1] <= 'Z'))) {
        word[len-1] = '\0';
        len--;
    }
    // Remove leading punctuation (shift)
    int start = 0;
    while (word[start] != '\0' && !((word[start] >= 'a' && word[start] <= 'z') || (word[start] >= 'A' && word[start] <= 'Z'))) {
        start++;
    }
    if (start > 0) {
        memmove(word, word + start, len - start + 1);
    }
}

// Function to read words from a file
int readWordsFromFile(const char *filename, char words[MAX_WORDS][50]) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    int count = 0;
    char buffer[100];
    while (fscanf(file, "%99s", buffer) == 1) {
        stripPunctuation(buffer);
        if (strlen(buffer) == 0) continue;
        
        // Truncate if too long (though strip might have shortage'd it)
        if (strlen(buffer) > MAX_WORD_LENGTH) {
            buffer[MAX_WORD_LENGTH] = '\0';
        }

        if (count >= MAX_WORDS) {
            fprintf(stderr, "Error: Maximum word limit reached\n");
            break;
        }
        strcpy(words[count], buffer);
        count++;
    }
    fclose(file);
    return count;
}

// Function to read subset words from a CSV file
// Function to read subset words from a CSV file
int readSubsetWordsFromCSV(const char *filename, char subsetWords[MAX_WORDS][50]) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    int count = 0;
    char line[100]; // Buffer to store each line from the CSV file
    // Skip the header line (first line)
    fgets(line, sizeof(line), file);
    while (fgets(line, sizeof(line), file)) {
        // Remove newline character at the end of the line
        line[strcspn(line, "\r\n")] = '\0'; // Handle both \n and \r\n
        // Skip empty lines
        if (strlen(line) == 0) {
            continue;
        }
        
        // Check for duplicates
        bool exists = false;
        for (int i = 0; i < count; i++) {
            if (strcmp(subsetWords[i], line) == 0) {
                exists = true;
                break;
            }
        }
        if (exists) continue;

        // Copy the word into the subsetWords array
        if (count < MAX_WORDS) {
            strncpy(subsetWords[count], line, 49);
            subsetWords[count][49] = '\0'; // Ensure null-termination
            count++;
        } else {
            fprintf(stderr, "Error: Maximum word limit reached\n");
            break;
        }
    }
    fclose(file);
    return count;
}

// Function to remove duplicate parent words
void removeDuplicateParents(char parentWords[MAX_WORDS][50], int *parentCount) {
    for (int i = 0; i < *parentCount; i++) {
        for (int j = i + 1; j < *parentCount; j++) {
            if (strcmp(parentWords[i], parentWords[j]) == 0) {
                // Shift remaining words to the left
                for (int k = j; k < *parentCount - 1; k++) {
                    strcpy(parentWords[k], parentWords[k + 1]);
                }
                (*parentCount)--; // Reduce the count of parent words
                j--; // Recheck the current index after shifting
            }
        }
    }
}

// Main function
int main() {
    char words[MAX_WORDS][50];
    int wordCount = readWordsFromFile("C:\\Users\\umesh\\Documents\\Programs\\Project5\\Source Files\\project_text.txt", words);
    Node *lists[MAX_WORD_LENGTH - MIN_WORD_LENGTH + 1] = {NULL}; // Array of linked list pointers

    // Insert words into linked lists
    for (int i = 0; i < wordCount; i++) {
        insertWord(lists, words[i]);
    }

    // Read subset words from a CSV file
    char subsetWords[MAX_WORDS][50];
    int subsetCount = readSubsetWordsFromCSV("C:\\Users\\umesh\\Documents\\Programs\\Project5\\Source Files\\project5_data.csv", subsetWords);

    // Display subset words, their parent words, and lengths
    printf("\nSubset Words and their Parent Words:\n");
    for (int i = 0; i < subsetCount; i++) {
        const char *subset = subsetWords[i];
        printf("\nSubset: %s (length: %lu)\n", subset, strlen(subset));
        char parentWords[MAX_WORDS][50];
        int parentCount = 0;

        // Find parent words for the current subset
        // Find parent words for the current subset using linked list buckets (Optimization)
        int startLen = strlen(subset);
        if (startLen < MIN_WORD_LENGTH) startLen = MIN_WORD_LENGTH;

        for (int len = startLen; len <= MAX_WORD_LENGTH; len++) {
             int listIndex = len - MIN_WORD_LENGTH;
             if (listIndex > (MAX_WORD_LENGTH - MIN_WORD_LENGTH)) break; // Safety check

             Node *current = lists[listIndex];
             while (current) {
                 if (containsAllCharacters(subset, current->word)) {
                     if (parentCount < MAX_WORDS) {
                         strncpy(parentWords[parentCount], current->word, 49);
                         parentWords[parentCount][49] = '\0';
                         parentCount++;
                     }
                 }
                 current = current->next;
             }
        }

        // Remove duplicate parent words
        removeDuplicateParents(parentWords, &parentCount);

        // Print the unique parent words
        for (int j = 0; j < parentCount; j++) {
            printf("  - Parent: %s (length: %lu)\n", parentWords[j], strlen(parentWords[j]));
        }
        printf("Total Parent Words: %d\n", parentCount);
    }

    // Free the linked lists
    freeLists(lists);
    return 0;
}