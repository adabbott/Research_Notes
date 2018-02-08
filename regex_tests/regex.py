"""
A testbed for python regex tech
"""
import re

# save some common regex features as human readable variables
letter           = r'[a-zA-Z]' 
unsigned_integer = r'\d+'      
double           = r'-?\d+\.\d+'
whitespace       = r'\s'       
endline          = r'\n'       
                               

def maybe(string):
    """
    A regex wrapper for an arbitrary string.
    Allows a string to be present, but still matches if it is not present.
    """
    return r'(?:{:s})?'.format(string)

def two_or_more(string):                 
  return r'(?:{:s}){{2,}}'.format(string)

print(maybe(letter))



# zero or more whitespace followed by x 
ws_double  = r"\s*" + double 
ws_endline = r"\s*" + endline 

# an xyz style geometry line, atom  x   y  z
geom_line_regex = r"\s*" + letter + maybe(letter) + 3 * ws_double + ws_endline
# an xyz style geometry block
geom_block_regex = two_or_more(geom_line_regex)

# test regex identifier for robustness
xyz1 = " C  0.0000 2.000 1.000  \n Cl  2.0000 1.000 0.000 \n BR  1.0000 0.000 2.000"

if re.match(geom_block_regex, xyz1):
    print('Yes')
else:
    print('No')


