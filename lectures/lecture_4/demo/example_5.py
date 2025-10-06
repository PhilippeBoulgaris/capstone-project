def func(a,b=2,*args,c,d=4,e,**kwargs):
  print("### positional arguments ###################")
  print(a)
  print("### positional arguments with defaults #####")
  print(b)
  print("### variadic positional arguments ##########")
  for (ia,a) in enumerate(args):
    print(ia+1,a)
  print("### keyword-only arguments #################")
  print(c)
  print(e)
  print("### keyword-only arguments with defaults ###")
  print(d)
  print("### variadic pkeyword-only arguments #######")
  for (ikwa,kwa) in enumerate(kwargs):
    print(ikwa+1,kwa)

func("a","b","pa","pb","pc",c=5,e=6,kwarg1="kwarg1",kwarg2="kwarg2")

