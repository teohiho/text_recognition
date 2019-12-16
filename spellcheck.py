# # import the fuzzywuzzy module
# from fuzzywuzzy import fuzz #https://github.com/seatgeek/fuzzywuzzy
#### https://github.com/pragnakalp/spellcheck-using-dictionary-in-python

# # def spellcheck(word_dict_file):


# # def readFile(word_dict_file):
# #     # open the dictionary file
# #     file = open(word_dict_file, 'r')
# #     # load the file data in a variable
# #     data = file.read()
# #     return data

# # def allWordsInList(data):
# #     # store all the words in a list
# #     data = data.split(",")
# #     # change all the words to lowercase
# #     data = [i.lower() for i in data]
# #     # remove all the duplicates in the list
# #     data = set(data)
# #     dictionary = list(data)
# #     return dictionary

# # spellcheck main class
# class SpellCheck:

#     # initialization method
#     def __init__(self, word_dict_file=None):
#         # open the dictionary file
#         self.file = open(word_dict_file, 'r')
        
#         # load the file data in a variable
#         data = self.file.read()
        
#         # store all the words in a list
#         data = data.split(",")
        
#         # change all the words to lowercase
#         data = [i.lower() for i in data]
        
#         # remove all the duplicates in the list
#         data = set(data)
        
#         # store all the words into a class variable dictionary
#         self.dictionary = list(data)

#     # string setter method
#     def check(self, string_to_check):
#         # store the string to be checked in a class variable
#         self.string_to_check = string_to_check

#     # this method returns the possible suggestions of the correct words
#     def suggestions(self):
#         # store the words of the string to be checked in a list by using a split function
#         string_words = self.string_to_check.split()

#         # a list to store all the possible suggestions
#         suggestions = []

#         # loop over the number of words in the string to be checked
#         for i in range(len(string_words)):
            
#             # loop over words in the dictionary
#             for name in self.dictionary:
                
#                 # if the fuzzywuzzy returns the matched value greater than 80
#                 if fuzz.ratio(string_words[i].lower(), name.lower()) >= 75:
                    
#                     # append the dict word to the suggestion list
#                     suggestions.append(name)

#         # return the suggestions list
#         return suggestions

#     # this method returns the corrected string of the given input
#     def correct(self):
#         # store the words of the string to be checked in a list by using a split function
#         string_words = self.string_to_check.split()

#         # loop over the number of words in the string to be checked
#         for i in range(len(string_words)):
            
#             # initiaze a maximum probability variable to 0
#             max_percent = 0

#             # loop over the words in the dictionary
#             for name in self.dictionary:
                
#                 # calulcate the match probability
#                 percent = fuzz.ratio(string_words[i].lower(), name.lower())
                
#                 # if the fuzzywuzzy returns the matched value greater than 80
#                 if percent >= 75:
                    
#                     # if the matched probability is
#                     if percent > max_percent:
                        
#                         # change the original value with the corrected matched value
#                         string_words[i] = name
                    
#                     # change the max percent to the current matched percent
#                     max_percent = percent
        
#         # return the cprrected string
#         return " ".join(string_words)        






# ################################################
import math
# https://www.slideshare.net/amrelarabi1/python-spell-checker
# https://github.com/dwyl/english-words
class SpellCheck:
    def __init__(self, word_dict_file=None):
        # open the dictionary file
        self.file = open(word_dict_file, 'r')
        
        # load the file data in a variable
        data = self.file.read()
        
        # store all the words in a list
        # data = data.split(",")
        data = data.split('\n')
        
        # change all the words to lowercase
        data = [i.lower() for i in data]
        
        # remove all the duplicates in the list
        data = set(data)
        
        # store all the words into a class variable dictionary
        self.wordlist = list(data)

    # Giống nhau độ dài hoặc chỉ khác nhau 1 đơn vị độ dài thì đêm đi so sánh nà
    def __areInRange(num1, num2, Range=1):
        for i in range(Range + 1):
            if(math.fabs(num1 - num2) == i):
                return True
        return False

    # So sánh từ đó với từ trong từ điển. Nếu cùng 1 vị trí mà giống nhau thì +1
    def __LettersInplace(word, Input):
        lettersInplace = 0
        for i in range(len(Input)):
            if(len(word) > i) and (word[i] == Input[i]):
                lettersInplace += 1
        return lettersInplace
    
    def __CommonLetters(word, Input):
        commonLetters = 0

        Inputs = {}
        for i in range(len(Input)):
            if(Inputs.get(Input[i], False)):
                Inputs[Input[i]] += 1
            else:
                Inputs[Input[i]] = 1
        
        for key, value in Inputs.items():
            if word.count(key) == value:
                commonLetters += 1
        
        if(Input[len(Input) - 1 ] == word[len(word) - 1]):
            commonLetters += 1
        if(len(Input) == len(word)):
            commonLetters += 1
        if(word[0] == Input[0]):
            commonLetters += 1
        return commonLetters

    def __LettersInRange(word, Input):
        lettersInRange = 0
        for i in range(len(Input)):
            if(len(word) > i) and (Input[i] in word) and SpellCheck.__areInRange(i, Input.index(word[i]) if(word[i] in Input) else -1):
                lettersInRange += 1
            return lettersInRange

    def IsValid(self, Input): 
        return Input in self.__tree


    def GetSuggestions(self, Input, NumOfSuggestions = 5): 
        arr = [] 
        for word in self.wordlist: 
            if SpellCheck.__areInRange(len(word), len(Input), 1): 
                arr.append(word) 
        
        dic = {} 
        for word in arr: 
            NumOfSimilarities = 0 
            NumOfSimilarities += SpellCheck.__CommonLetters(str(word), Input) 
            NumOfSimilarities += SpellCheck.__LettersInplace(str(word), Input) 
            NumOfSimilarities += SpellCheck.__LettersInRange(str(word), Input)
            dic[str(word)] = NumOfSimilarities
            
        Maxes = []
        for i in range(NumOfSuggestions):
            if len(dic) > 0:
                Max = list(dic.keys())[list(dic.values()).index(max(dic.values()))]
                Maxes.append(Max)
                del dic[str(Max)]
                
        return Maxes


