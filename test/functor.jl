using TumorGrowth
using ComponentArrays

nt = (x = 1, y =2)
c = ComponentArray(nt)
nt2, reconstruct = TumorGrowth.functor(c)
@test nt == nt2
@test reconstruct(nt2) == c

true
