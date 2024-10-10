'''
This script is about cleaning text using "re module" in Python.
Regular expressions are patterns used to match and manipulate text.
With re, we can search for specific patterns, replace text, or split
strings based on those patterns.

Characters in a Set
    [a-z] matches lowercase letters
    [A-Z] matches uppercase letters
    [a-zA-Z] matches lowercase and uppercase letters.

'Lookbehind' Assertion

A 'lookbehind' assertion in regular expressions allows you to check for a pattern that precedes another pattern without including it in the match result.

How It Works
Syntax:

(?<=...) is the syntax for a positive lookbehind.
(?<!...) is the syntax for a negative lookbehind.
Positive Lookbehind ((?<=...)):

Checks if a certain pattern precedes another pattern.
Example: (?<=@)\w+ matches a word only if it is preceded by @, like @username but matches only username.
Negative Lookbehind ((?<!...)):

Checks if a certain pattern does not precede another pattern.
Example: (?<!@)\w+ matches a word only if it is not preceded by @.

'Lookahead' Assertion
A 'lookahead' assertion in regular expressions is used to check for a pattern that follows another pattern without including it in the match result.

How It Works
Syntax:

(?=...) is the syntax for a positive lookahead.
(?!...) is the syntax for a negative lookahead.
Positive Lookahead ((?=...)):

Checks if a certain pattern follows another pattern.
Example: \d(?=\D) matches a digit only if it is followed by a non-digit character.
Negative Lookahead ((?!...)):

Checks if a certain pattern does not follow another pattern.
Example: \d(?!\d) matches a digit only if it is not followed by another digit.

You can do this by using the lookahead assertion. (?=...)
Here is the documentation:

Matches if ... matches next, but doesn’t consume any of the string. This is called a lookahead assertion. For example, Isaac (?=Asimov) will match 'Isaac ' only if it’s followed by 'Asimov'.

this: (?=[a-zA-Z])
'''

import re

# Search patterns
# Search if string starts with 'Pl' or ends with 'ion'
match1 = re.search(pattern="^Pl|ion$", string="Pleural Effusion")

# Return the matched string
print(match1.group(0) if match1 else None)

# Search if string starts with 'Sa' or ends with 'ion'
match2 = re.search(pattern="^Sa|ion$", string="Pleural Effusion")
print(match2.group(0) if match2 else None)

# Search if string starts with 'Eff'
match3 = re.search(pattern="^Eff", string="Pleural Effusion")
print(match3.group(0) if match3 else None)

# Define a pattern with groups
pattern = r"(\d{3})-(\d{2})-(\d{4})"  # Matches a phone number in the format xxx-xx-xxxx

# Search for the pattern in a string
match = re.search(pattern, "My number is 123-45-6789")

if match:
    print("Full match:", match.group(0))  # Prints: 123-45-6789
    print("First group:", match.group(1)) # Prints: 123
    print("Second group:", match.group(2)) # Prints: 45
    print("Third group:", match.group(3)) # Prints: 6789
else:
    print("No match found")

# .group(0) is useful for retrieving the complete match when you want to know what part of the string fits the entire pattern.

# Match a single letter of the alphabet followed by a number
match4 = re.search(pattern='[a-zA-Z]123', string="99C123")
print(match4.group(0))

# 'Lookbehind' Assertion
# Match a letter followed by 123, exclude the letter
match5 = re.search(pattern='(?<=[a-zA-Z])123', string="99C123")
print(match5.group(0))

# Match 123 followed by a letter
match6 = re.search(pattern='123[a-zA-Z]', string="99123C99")
print(f"{match6.group(0)}")

# 'Lookahead' Assertion
# Match 123 followed by a letter, exclude the letter from the returned match
match7 = re.search(pattern='123(?=[a-zA-Z])', string="99123C99")
print(f"{match7.group(0)}")

# String Cleaning
# Choose a sentence to be cleaned
sentence = "     BIBASILAR OPACITIES,likely representing bilateral pleural effusions with ATELECTASIS   and/or PNEUMONIA/bronchopneumonia.."
print(sentence)

# Convert to lowercase only
sentence = sentence.lower()
print(sentence)

# Change "and/or" to "or"
sentence = re.sub('and/or', 'or', sentence)
print(sentence)

# Change "/" to "or"
sentence = re.sub('(?<=[a-zA-Z])/(?=[a-zA-Z])', ' or ', sentence)
print(sentence)

# Replace .. with . using re.sub
tmp1 = re.sub("\.\.", ".", sentence) # option 1
# print(tmp1)

tmp2 = sentence.replace('..', '.') # option 2
# print(tmp2)

sentence = sentence.replace('..', '.')
print(sentence)

# Add Whitespace after Punctuation