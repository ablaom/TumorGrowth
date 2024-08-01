using TumorGrowth
using ComponentArrays

nt = (x = 1, y =2)
c = ComponentArray(nt)
nt2, reconstruct = TumorGrowth.functor(c)
@test nt == nt2
@test reconstruct(nt2) == c

t = (x = 1, y =2, z=3, w=4)
c = ComponentArray(t)
frozen = (w=40, y=20)
cfree, reconstruct = TumorGrowth.functor(c, frozen)
@test cfree == (x=1, z=3)

t2 = (x = 100, y =200, z=300, w=400)
c2 = ComponentArray(t2)
@test reconstruct(t2) == ComponentArray((x=100, y=20, z=300, w=40))
@test reconstruct(c2) == ComponentArray((x=100, y=20, z=300, w=40))

true
