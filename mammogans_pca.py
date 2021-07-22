import torch 
import torch.tensor as tensor
import pickle as pkl
from sklearn.decomposition import PCA

with open("latents.pkl", "rb") as file:
        latents_set = pkl.load(file)

pca = PCA(10)
latents_processed = latents_set.reshape(16484,16*512).cpu()
latents_processed = latents_processed/latents_processed.std(0).cpu()
pc = pca.fit_transform(latents_processed)
components = tensor(pca.components_.reshape(10,16,512)).cuda()
mean = latents_set.mean(0)
std = latents_set.std(0)

result = [components,mean,std,pca.explained_variance_]

with open("pca_result.pkl", "wb") as fout:
        pkl.dump(result, fout, protocol=pkl.HIGHEST_PROTOCOL)


# Test
a = (latents_set[0]-mean)/std
b = (a*components[0]).sum()
print(b-tensor(pc[0][0]))