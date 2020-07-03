import sys,os,eel
eel.init('web')
session=0
dataPath=os.getcwd()+"/Data/"

@eel.expose
def loginpy(username,password):
    global session
    if(username=="admin@sih.com" and password=="1234"):
        session=1
        return session
    else:
        return session

@eel.expose
def logoutpy():
    global session
    session=0


@eel.expose
def checkSessionpy():
    global session
    return session

@eel.expose  
def trainingpy():
    import trainning
    return "completed"

@eel.expose
def facerecogpy():
    import vedio

@eel.expose
def timestamppy(ID_NUMBER):
    path=dataPath+"timestamps/"+str(ID_NUMBER)+'.txt'
    os.startfile(path)

@eel.expose
def photopy(ID_NUMBER):
    path=dataPath+"photos/"+str(ID_NUMBER)+'.jpg'
    os.startfile(path)

@eel.expose
def Dataset(id,name):
    from dataset import det
    det(id,name)
    return "completed"

eel.start('login.html')