
  
import pickle
with open ("C:\\Users\\z5021\\OneDrive\\Desktop\\Game\\MLGame-master\\games\\arkanoid\\log\\test1\\2019-10-02_16-44-00.pickle" , "rb") as f1:
	data_list1 = pickle.load(f1)
with open ("C:\\Users\\z5021\\OneDrive\\Desktop\\Game\\MLGame-master\\games\\arkanoid\\log\\test1\\2019-10-08_19-29-15.pickle" , "rb") as f2:
	data_list2 = pickle.load(f2)
with open ("C:\\Users\\z5021\\OneDrive\\Desktop\\Game\\MLGame-master\\games\\arkanoid\\log\\test1\\2019-10-08_19-32-17.pickle" , "rb") as f3:
	data_list3 = pickle.load(f3)
with open ("C:\\Users\\z5021\\OneDrive\\Desktop\\Game\\MLGame-master\\games\\arkanoid\\log\\test1\\2019-10-08_20-37-32.pickle" , "rb") as f4:
	data_list4 = pickle.load(f4)
with open ("C:\\Users\\z5021\\OneDrive\\Desktop\\Game\\MLGame-master\\games\\arkanoid\\log\\test1\\2019-10-08_20-41-02.pickle" , "rb") as f5:
	data_list5 = pickle.load(f5)
with open ("C:\\Users\\z5021\\OneDrive\\Desktop\\Game\\MLGame-master\\games\\arkanoid\\log\\test1\\2019-10-08_20-42-40.pickle" , "rb") as f6:
	data_list6 = pickle.load(f6)
with open ("C:\\Users\\z5021\\OneDrive\\Desktop\\Game\\MLGame-master\\games\\arkanoid\\log\\test1\\2019-10-08_20-44-46.pickle" , "rb") as f7:
	data_list7 = pickle.load(f7)






Frame = [ ]
Status = [ ]
Ballposition = [ ]
Platformposition = [ ]
Bricks = [ ]
for i in range (0,len(data_list1)):
	Frame.append(data_list1[i].frame)
	Status.append(data_list1[i].status)
	Ballposition.append(data_list1[i].ball)
	Platformposition.append(data_list1[i].platform)
	Bricks.append(data_list1[i].bricks)
for i in range (0,len(data_list2)):
	Frame.append(data_list2[i].frame)
	Status.append(data_list2[i].status)
	Ballposition.append(data_list2[i].ball)
	Platformposition.append(data_list2[i].platform)
	Bricks.append(data_list2[i].bricks)
for i in range (0,len(data_list3)):
	Frame.append(data_list3[i].frame)
	Status.append(data_list3[i].status)
	Ballposition.append(data_list3[i].ball)
	Platformposition.append(data_list3[i].platform)
	Bricks.append(data_list3[i].bricks)
for i in range (0,len(data_list4)):
	Frame.append(data_list4[i].frame)
	Status.append(data_list4[i].status)
	Ballposition.append(data_list4[i].ball)
	Platformposition.append(data_list4[i].platform)
	Bricks.append(data_list4[i].bricks)
for i in range (0,len(data_list5)):
	Frame.append(data_list5[i].frame)
	Status.append(data_list5[i].status)
	Ballposition.append(data_list5[i].ball)
	Platformposition.append(data_list5[i].platform)
	Bricks.append(data_list5[i].bricks)
for i in range (0,len(data_list6)):
	Frame.append(data_list6[i].frame)
	Status.append(data_list6[i].status)
	Ballposition.append(data_list6[i].ball)
	Platformposition.append(data_list6[i].platform)
	Bricks.append(data_list6[i].bricks)
for i in range (0,len(data_list7)):
	Frame.append(data_list7[i].frame)
	Status.append(data_list7[i].status)
	Ballposition.append(data_list7[i].ball)
	Platformposition.append(data_list7[i].platform)
	Bricks.append(data_list7[i].bricks)




#----------------------------------------------------------------------------------------------------------------------------
import numpy as np
PlatX = np.array(Platformposition)[:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]
instruct = (PlatX_next - PlatX[0:len(PlatX_next),0][:, np.newaxis])/5

BallX=np.array(Ballposition)[:,0][:,np.newaxis]
BallX_next=BallX[1:,:]
vx=(BallX_next-BallX[0:len(BallX_next),0][:,np.newaxis])

BallY=np.array(Ballposition)[:,1][:,np.newaxis]
BallY_next=BallY[1:,:]
vy=(BallY_next-BallY[0:len(BallY_next),0][:,np.newaxis])

Ballarray = np.array(Ballposition[:-1])

x = np.hstack((Ballarray , PlatX[0:-1,0][:,np.newaxis],vx,vy))
#x = np.hstack((Ballarray , PlatX[0:-1,0][:,np.newaxis]))
y = instruct

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=41)

#----------------------------------------------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train,y_train)

yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler, y_test)


#----------------------------------------------------------------------------------------------------------------------------
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(x_train)
#x_train_stdnorm = scaler.transform(x_train)
#knn.fit(x_train_stdnorm ,y_train)
#x_test_stdnorm = scaler.transform(x_test)
#yknn_aft_scaler = knn.predict(x_test_stdnorm)
#acc_knn_aft_scaler = accuracy_score(yknn_aft_scaler,y_test)


#----------------------------------------------------------------------------------------------------------------------------
filename = "C:\\Users\\z5021\\OneDrive\\Desktop\\Game\\MLGame-master\\games\\arkanoid\\log\\test1\\KNN.sav"
pickle.dump(knn , open(filename , 'wb'))

