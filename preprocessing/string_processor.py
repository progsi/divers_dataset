import re 
from unidecode import unidecode

class StringProcessor:
    def __init__(self, exclude: str = "'"):
        self.exclude = exclude
        self.apostrophes = "'’‘`´ʻʼʽ"
    
    @staticmethod
    def unidecode_letters(s: str) -> str:
        def replace_with_unidecode(match):
            char = match.group(0)
            return unidecode(char)
        s = re.sub(r'[^\W\d_]', replace_with_unidecode, s)
        return s

    @staticmethod
    def isolate_special_chars(s: str, exclude: str = "'") -> str:
        """Separates special chars.
        Args:
            s (str): input string
            exclude (str): string of chars to not isolate (typically ')
        Returns:
            str: string with isolated special chars
        """
        special_chars = r'([!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~])'.replace(exclude, "")
        s = re.sub(special_chars, r' \1 ', s)
        s = re.sub(r'\s+', ' ', s)
        s = s.strip()
        return s

    def __call__(self, s: str) -> str:
        """Only retain space, latin chars and numbers. Remove attached special chars
        Args:
            s (str): 
        Returns:
            str: 
        """
        s = s.strip().lower()
        # remove apostrophe
        for a in self.apostrophes:
            s = s.replace(a, "")

        # to basic latin 
        s = ' '.join([self.unidecode_letters(w).replace(" ", "") for w in s.split()])
        # isolation
        s = self.isolate_special_chars(s)
        return s
