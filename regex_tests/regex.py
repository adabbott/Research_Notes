"""
A testbed for python regex tech
"""
import re

# save some common regex features as human readable variables
letter           = r'[a-zA-Z]' 
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
    """
    A regex wrapper for an arbitrary string.
    Allows multiple instances of the string to be matched.
    """
    return r'(?:{:s}){{2,}}'.format(string)

# zero or more whitespace (ws) followed by the regex feature 
ws_double  = r"\s*" + double    # for floats
ws_endline = r"\s*" + endline   # for newlines 

# a regex identifier for an xyz style geometry line, atom_label  x_coord   y_coord  z_coord
geom_line_regex = r"[ \t]*" + letter + maybe(letter) + 3 * ws_double + ws_endline
# an xyz style geometry block
geom_block_regex = two_or_more(geom_line_regex)
print(geom_block_regex)

# test regex identifier with xyz string
xyz1 = " C  0.0000 2.000 1.000  \n Cl  2.0000 1.000 0.000 \n Br  1.0000 0.000 2.000 \n"

if re.match(geom_block_regex, xyz1):
    print('Yes')
else:
    print('No')


# test pulling from a sample input file and writing new input file
with open('input.dat', 'r') as f:
    data = f.read()

new = re.sub(geom_block_regex, xyz1, data)

with open('input2.dat', 'w+') as f:
    f.write(new)
