def fprint(*args) :
	var=args[len(args)-1]
	for i in range(len(args)-1) :
		print(args[i], end ="")

	print("{0}: {1}".format(var, repr(eval(var))))



def test() :
	goat=12
	fprint("test", "testb", "goat")

test()
