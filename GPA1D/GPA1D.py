from GPA import GPA
import pandas as pd
import numpy as np
import gilbert

class GPA1D():
	def __init__(self,tol=0.03, spaceFilling='lines', splitWidth=3):
		'''
			tol - muduli tolerance
			spaceFilling - spatial curve to transform the time series into matrices ('lines','hilbert') 
			scale - lattice size generated for spacefilling curve
		'''
		self.ga = GPA(tol)
		self.splitWidth = splitWidth
		self.spaceFilling = spaceFilling
	
	def verifyPower2(self,value):
		i = int(np.log2(value))//2
		while (i>=1):
			if value % (2**i) > 0:
				return False
			i= i-1
		return True

	def _transformData(self,vet):
		if self.spaceFilling == 'hilbert':
			mat = gilbert.vec2mat(vet, self.splitWidth)
		elif self.spaceFilling == 'lines':
			mat = vet.reshape(self.splitWidth,self.splitWidth)
			for i in range(1,len(mat),2):
				mat[i] = np.flip(mat[i])
		else:
			mat = vet.reshape(self.splitWidth,self.splitWidth)
		return mat

	def __call__(self,timeSeries,moment=["G2","G3"],symmetrycalGrad='A'):
		#if len(timeSeries) % (self.splitWidth**2) != 0:
		#	raise Exception(f"The time series must be multiple of {self.splitWidth**2}. You can interpolate or remove some elements.")
		#splitted = np.array_split(timeSeries, len(timeSeries) // (self.splitWidth**2))
		res = []
		for i in range(0,len(timeSeries)-self.splitWidth**2,self.splitWidth**2):
			s =  timeSeries[i:i+self.splitWidth**2]
			mat = np.array(self._transformData(s)).astype(float)
			res.append(self.ga(mat,moment=['G1','G2','G3','G4'] ))
		return pd.DataFrame(res)
	
	def getGa(self):
		return self.ga
		
	def scalingLaw(timeSeries,moment='G1', returnFit=False,start=3,stop=17,step=2,spaceFilling='hilbert'):
		pts = []
		for sw in range(start,stop,step):
			ga = GPA1D(splitWidth=sw,spaceFilling=spaceFilling)
			res = ga(timeSeries)
			res['scale'] = sw
			pts.append(res)

		pts = pd.concat(pts, ignore_index=True)
		pts = pts.groupby('scale',as_index=False).mean()
		fit = np.polyfit(np.log(pts['scale']), np.log(pts[moment]),1)
		if returnFit:
			return fit[0], pts, np.exp(np.polyval(fit, np.log(pts['scale'])))
		else:
			return fit[0]
