3c. pairs.txt format
----------------

The pairs.txt file is formatted as follows: The top line gives the
number of sets followed by the number of matched pairs per set (equal
to the number of mismatched pairs per set).  The next 300 lines give
the matched pairs in the following format:

name   n1   n2

which means the matched pair consists of the n1 and n2 images for the
person with the given name.  For instance,

George_W_Bush   10   24

would mean that the pair consists of images George_W_Bush_0010.jpg and
George_W_Bush_0024.jpg.

The following 300 lines give the mismatched pairs in the following format:

name1   n1   name2   n2

which means the mismatched  pair consists of  the n1 image  of person
name1 and the n2 image of person name2.  For instance,

George_W_Bush   12   John_Kerry   8

would mean that the pair consists of images George_W_Bush_0012.jpg and
John_Kery_0008.jpg.

This procedure is then repeated 9 more times to give the pairs for the
next 9 sets.