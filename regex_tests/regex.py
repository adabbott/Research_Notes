"""
A testbed for python regex tech
"""
import re

# save some common regex features as human readable variables
letter           = r'[a-zA-Z]' 
double           = r'-?\d+\.\d+'
integer          = r'\d+'
whitespace       = r'\s'       
endline          = r'\n'       
                               

def maybe(string):
    """
    A regex wrapper for an arbitrary string.  Allows a string to be present, but still matches if it is not present.
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

# test regex identifier with xyz string
xyz1 = " C  0.0000 2.000 1.000  \n Cl  2.0000 1.000 0.000 \n Br  1.0000 0.000 2.000 \n"


# test pulling from a sample input file and writing new input file
with open('input.dat', 'r') as f:
    data = f.read()

print(data)

#print(re.match(geom_block_regex, data))
#new = re.sub(geom_block_regex, xyz1, data)
#
#with open('input2.dat', 'w+') as f:
#    f.write(new)

#xyz_block_regex = geom_block_regex
#
## if there is more than one xyz, take the last one
#iter_matches = re.finditer(xyz_block_regex, data, re.MULTILINE)
#matches = [match for match in iter_matches]
#if matches is None:
#    raise Exception("No XYZ geometry found in template input file")
#
#start = matches[-1].start()
#end   = matches[-1].end()
#
#xyz = data[start:end]


#########################
# begin attempt to parse compact internal coordinates
#########################
"""
O
H 1 1.0
H 1 1.0 2 104.5
H 1 1.0 2 100.00 3 180.0
"""

integer    = r'\d+'
ws_int     = r"\s*" + integer  # for integers

int_line1_regex = letter + maybe(letter) + ws_endline
int_line2_regex = letter + maybe(letter) + ws_int + ws_double + ws_endline
int_line3_regex = letter + maybe(letter) + ws_int + ws_double + ws_int + ws_double + ws_endline
int_line4_regex = letter + maybe(letter) + ws_int + ws_double + ws_int + ws_double + ws_int + ws_double + ws_endline

def one_or_more(string):                 
    """
    A regex wrapper for an arbitrary string.
    Allows multiple instances of the string to be matched.
    """
    return r'(?:{:s}){{1,}}'.format(string)

test_regex = int_line1_regex + int_line2_regex + maybe(int_line3_regex) + maybe(one_or_more(int_line4_regex))

print(re.search(test_regex, data))


