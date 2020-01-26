# Permutation-Generators

Here we go over a few different mechanisms for efficiently generating permutations with a fixed pinnacle set.

- `magicPinGen`, pending name change

### Breakdown of methods

###### Main Methods:

- `magicPinGenCR`

- `magicPinGenFull`

###### Other Methods:

- `decompSeq`
  - produces a decomposition sequence for a permutation

- `orderedDecompSeq`
  - like decompSeq, but performs decomposition in order

### Examples:

`>>> orderedDecompSeq([0, 5, 4, 1, 3, 2])`<br/>
`[[(0, 5, 4, 1, 3, 2)], [(0,), (4, 1, 3, 2)], [(0,), (1, 3, 2)],`<br/>
&nbsp; `[(0,), (1,), (2,)], [(0,), (1,)], [(0,)], []]`

`>>> orderedDecompSeq([0, 1, 3, 2])`<br/>
`[[(0, 1, 3, 2)], [(0, 1), (2,)], [(0, 1)], [(0,)], []]`

`>>> [*magicPinGenFull(4, {3})]`<br/>
`[(0, 3, 1, 2), (1, 2, 3, 0), (0, 3, 2, 1), (2, 1, 3, 0),`<br/>
&nbsp; `(1, 3, 0, 2), (0, 2, 3, 1), (1, 3, 2, 0), (2, 0, 3, 1),`<br/>
&nbsp; `(0, 1, 3, 2), (2, 3, 0, 1), (1, 0, 3, 2), (2, 3, 1, 0)]`

`>>> [*magicPinGenCR(4, {3})]`<br/>
`[(0, 3, 1, 2), (1, 3, 0, 2), (0, 1, 3, 2)]`