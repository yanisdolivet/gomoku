/**
 * This file is a configuration file for commitlint
 * It is used to define the rules for commit messages
 */

module.exports = {
    parserPreset: {
        parserOpts: {
            // Regex to capture : [TYPE] Message
            headerPattern: /^\[(\w+)\] (.+)$/,
            headerCorrespondence: ['type', 'subject'],
        },
    },
    rules: {
        // List of allowed types (in MAJUSCULES)
        'type-enum': [
            2,
            'always',
            [
                'ADD',
                'FIX',
                'UPDATE',
                'REMOVE',
                'DOC',
                'REFACTOR',
                'TEST',
                'CI',
                'MERGE',
                'REBASE'
            ],
        ],
        // The type must be in uppercase (ex: ADD, not add)
        'type-case': [2, 'always', 'upper-case'],
        // The subject must not be empty
        'subject-empty': [2, 'never'],
        // No period at the end of the title (good practice)
        'header-full-stop': [2, 'never', '.'],
        // Maximum length of the title (good practice)
        'header-max-length': [2, 'always', 72],
    },
};