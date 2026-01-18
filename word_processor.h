#ifndef WORD_PROCESSOR_H
#define WORD_PROCESSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_WORD_LENGTH 50
#define MIN_WORD_LENGTH 4
#define MAX_WORDS 1000

// Define the linked list node structure
typedef struct Node {
    char word[MAX_WORD_LENGTH + 1]; // Word storage
    struct Node *next;
} Node;

// Function prototypes
Node* createNode(const char *word);
bool isAlphabetic(const char *word);
void insertWord(Node **lists, const char *word);
void freeLists(Node **lists);
bool containsAllCharacters(const char *subset, const char *parent);
int readWordsFromFile(const char *filename, char words[MAX_WORDS][50]);
int readSubsetWordsFromCSV(const char *filename, char subsetWords[MAX_WORDS][50]);
void removeDuplicateParents(char parentWords[MAX_WORDS][50], int *parentCount);

#endif // WORD_PROCESSOR_H