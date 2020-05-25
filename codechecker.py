programfile = open('/root/mlops/mlproject.py','r')	
code = programfile.read()				

if 'keras' or 'tensorflow' in code:						
	if 'Conv2D' in code:				
		print('This is CNN')
	else:
		print('This is not CNN')
else:
	print('This is not deep learning')
