import vtk
import numpy as np
import pandas as pd
import math
import os 

import tkinter as tk 
import tkinter.filedialog 

#------------------------------------ Read Classification File -------------------------
def get_class(filename,Master_list):
    f = open(filename,'r')
    clsfct = {}
    label = set()
    f.readline()
    fl = f.readline()[:-1].split('\n')
    fl = fl[0].split(',')
    i = 0
    # print(Master_list)
    while (fl != ['']):
        if str(int(float(fl[0]))) in Master_list:
            fl = f.readline().split('\n')
            fl = fl[0].split(',')
            continue
        clsfct[i] = int(fl[1])
        label.add(int(fl[1]))
        fl = f.readline().split('\n')
        fl = fl[0].split(',')
        i += 1
        
    num_label = len(label)   
    return clsfct , num_label   

    
#------------------------------------ Read v3d File ------------------------------------
def get_data_v3d(filename, part_num):
    f = open(filename,'r')
    data = ['BOUNDARIES','0','CircularBOUNDARIES','0','PARTICLES']
    f.readline()
    numParticle = f.readline()[:-1]
    data = data + [numParticle]
    fl = f.readline()
    Master_list = []
    skip_particle = 0
    save_particle = 0
    while True:
        if fl == 'ITEMS: PARTICLES\n':
            break
        if fl == 'Element: id mass type centriod[3] minertia[3] princpdir[3][3] velocity[2][3] vertex[n][3]\n':
            vertex_new = []
            id_mass_type = f.readline()[:-1].split(',')
            shape = int(id_mass_type[2])-1
            part_id = int(id_mass_type[0])
            if shape == 0:
                fl = f.readline()
                Master_list.append(id_mass_type[0])
                while (fl != 'Element: id mass type centriod[3] minertia[3] princpdir[3][3] velocity[2][3] vertex[n][3]\n'):
                    if fl == 'ITEMS: PARTICLES\n':
                        break
                    fl = f.readline()
            elif part_id not in part_num.values:
#                 print(part_id)
                fl = f.readline()
                skip_particle += 1
                while (fl != 'Element: id mass type centriod[3] minertia[3] princpdir[3][3] velocity[2][3] vertex[n][3]\n'):
                    if fl == 'ITEMS: PARTICLES\n':
                        break
                    fl = f.readline()
            else:  
#                 print(part_id)
                save_particle += 1
                for i in range(7):
                    f.readline()
                fl = f.readline()
                while (fl != 'Element: id mass type centriod[3] minertia[3] princpdir[3][3] velocity[2][3] vertex[n][3]\n'):
                    if fl == 'ITEMS: PARTICLES\n':
                        break
                    vertex_new += fl[:-1].split(',')
                    fl = f.readline()
                data += [str(int(len(vertex_new)/3)),str(shape)] + vertex_new
    data[5] = str(int(data[5])-len(Master_list)-skip_particle)
    f.close()
    return data, Master_list

def visualization(library,data,clsfct,num_label):
#------------------------------------ Read Plot File ------------------------------------
    idx =0
    total_corner =0
    Total_circular_corner=0
    
    while(idx <len(data)-1):
        if(data[idx]=='BOUNDARIES'):
            idx =idx+1
            numBoundary = int(data[idx])
            idx =idx+1
            Boundary =  np.zeros((numBoundary, 4*3)) #4 corners of each Boundary
            bd_total_corner = numBoundary*4
            option = 'Boundary'
            
        elif(data[idx] =='PARTICLES'):
            idx =idx+1
            numParticle = int(data[idx])
            idx =idx+1
            Particle = [] #number of Particles
            numCorner =np.zeros(numParticle)#number of corners for the Particle
            accumulateCornerNum =[]
            shape =[]#Particle Library Number for each Particle
            option = 'Particle'
            
        elif(data[idx] =='CircularBOUNDARIES'):
            idx =idx+1
            numCircularBoundary = int(data[idx])
            idx =idx+1
            CircularBoundary = [] #number of Particles
            CircualrCorner =np.zeros((numCircularBoundary, 4)) #number of corners for the Particle
            Total_circular_corner = numCircularBoundary*20
            option = 'CircularBOUNDARIES'
    
        if(option == 'Boundary'):
            for i in range (0, numBoundary):# each Boundary
                for j in range (0, 4):#x y z of each corners
                    Boundary[i][j*3] =float(data[idx])
                    idx = idx+1
                    Boundary[i][j*3+1] = float(data[idx])
                    idx = idx+1
                    Boundary[i][j*3+2] = float(data[idx])
                    idx = idx+1
                    
        if(option == 'CircularBOUNDARIES'):
            for i in range (0, numCircularBoundary):# each Boundary
                CircualrCorner[i][0] =float(data[idx])
                idx = idx+1
                CircualrCorner[i][1] = float(data[idx])
                idx = idx+1
                CircualrCorner[i][2] = float(data[idx])
                idx = idx+1
                CircualrCorner[i][3] = float(data[idx])
                idx = idx+1  
                
        if(option == 'Particle'):
            total_corner = bd_total_corner+Total_circular_corner
            for i in range (0, numParticle): #for each particle
                size = int(data[idx])
                accumulateCornerNum.append(int(total_corner))
                total_corner = total_corner+size
                idx =idx+1
                #Particle[i] = np.zeros(size*3)#store number of corner x y z
                numCorner[i] =size #store number of corner
                shape.append(int(data[idx]))#store library number
                idx =idx+1
                for j in range (0, size):
                    Particle.append(float(data[idx]))
                    idx = 1+ idx
                    Particle.append(float(data[idx]))
                    idx = 1+ idx
                    Particle.append(float(data[idx]))
                    idx = 1+ idx
                    
    # store the vertex locations.
    vertex_locations = vtk.vtkPoints()
    vertex_locations.SetNumberOfPoints(total_corner)
    for nb in range (0, numBoundary):
        for i in range (0,4):
            vertex_locations.SetPoint(nb*4+i, (Boundary[nb][i*3],Boundary[nb][i*3+2], Boundary[nb][i*3+1]))
    
    setnumber = 4*numBoundary 
    for nb in range (0, numCircularBoundary):
        for i in range (0,20):
            vertex_locations.SetPoint(setnumber, (CircualrCorner[nb][0]+CircualrCorner[nb][3]*np.cos(math.radians(360/20*i)),CircualrCorner[nb][2], CircualrCorner[nb][1]+CircualrCorner[nb][3]*np.sin(math.radians(360/20*i))))
            setnumber = setnumber +1
    
    i =0
    while (setnumber < total_corner):
        x=i
        y=i+1
        z=i+2
        vertex_locations.SetPoint(setnumber, (Particle[x], Particle[z], Particle[y]))
        setnumber = setnumber +1
        i= i+3
    
    del data[:] 
    del Particle[:]
    
    lib= open("library.vlb",'r')
    #--------------------------------- Read Library -------------------------------
    libdata=[]
    for line in lib.read().split('\n'):
        if not line.startswith('//') and not line.startswith('Particle') and not line.startswith('Master'):
            for chunk in line.split(' '):
                for word in chunk.split('\t'):
                    if word=='':
                        continue
                    if word=='\t':
                        continue
                    libdata.append(word)
    
    lib.close()
    idx =0
    numParticleShape = int(libdata[idx])
    idx = idx+1
    numMasterShape = int(libdata[idx])
    idx = idx+1
    Particlelib=[]
    Particles_face = []
    for ns in range (0,numParticleShape):
        Particlelib.append([])
        skip1 = int(libdata[idx]) #num of points
        idx = idx+1
        face = int(libdata[idx]) #num of face
        Particles_face.append(face) #save num of faces for each particle in library
        idx = idx +1
        skip2 = int(libdata[idx]) #num of edge
        idx = idx+1
        idx = idx+skip1*3
        i =0
        
        while i < face:
            numface = int(libdata[idx]) # num of points on a face
            Particlelib[ns].append(numface)
            idx = idx +1
            i = i+1
            for nf in range (0, numface):
                Particlelib[ns].append(int(libdata[idx]))
                idx = idx +1
        for i in range (0, skip1):
            add = int(libdata[idx])
            idx = idx + add+1
        idx = idx+1
    
    numMas =0;   
    for ms in range (0,numMasterShape):
        numslave = int(libdata[idx])
        idx = idx +2
        for slav in range (0, numslave):
            Particlelib.append([])
            skip1 = int(libdata[idx])
            idx = idx+1
            face = int(libdata[idx])
            Particles_face.append(face) #save num of faces for each particle
            idx = idx +1
            skip2 = int(libdata[idx])
            idx = idx+1
            idx = idx+skip1*3
            i =0
            while i < face:
                numface =int(libdata[idx])
                Particlelib[numParticleShape+numMas].append(numface)
                idx = idx +1
                for nf in range (0, numface):
                    Particlelib[numParticleShape+numMas].append(int(libdata[idx]))
                    idx = idx +1
                i =i+1
                
            for i in range (0, skip1):
                add = int(libdata[idx])
                idx = idx + add+1
            idx = idx+1+skip1+3
            numMas = numMas+1
    del libdata[:] 
    # ----------------------------------- Set Face ----------------------------
    # describe the polygons that represent the faces using the vertex
    # indices in the vtkPoints that stores the vertex locations. There are a number
    # of ways to do this that you can find in examples on the Wiki.
    boundary_faces = vtk.vtkCellArray()
    
    for nb in range(0, numBoundary):
        if (nb!=0):
           continue
        q = vtk.vtkPolygon()
        q.GetPointIds().SetNumberOfIds(4) #make a quad
        q.GetPointIds().SetId(0, nb*4)
        q.GetPointIds().SetId(1, nb*4+1)
        q.GetPointIds().SetId(2, nb*4+2)
        q.GetPointIds().SetId(3, nb*4+3)
       
        boundary_faces.InsertNextCell(q)
    
    bd = vtk.vtkPolyData()
    bd.SetPoints(vertex_locations)
    bd.SetPolys(boundary_faces)
    
    Circular_boundary_faces = vtk.vtkCellArray()
    
    for nb in range(0, numCircularBoundary):
        cir = vtk.vtkPolygon()
        cir.GetPointIds().SetNumberOfIds(20) #make a quad
        for ed in range(0, 20):
            cir.GetPointIds().SetId(ed, nb*20 + ed)
    
        Circular_boundary_faces.InsertNextCell(cir)
    
    # Next create a vtkPolyData to store your face and vertex information that
    # represents your polyhedron.
    cbd = vtk.vtkPolyData()
    cbd.SetPoints(vertex_locations)
    cbd.SetPolys(Circular_boundary_faces)
    
    #-----------------------color generation------------------------------------
    color_list = [(255,0,0),(255,128,0),(255,255,0),(0,204,0),(0,204,204),(0,0,204),(102,0,204),(96,96,96)]
    
    # set up color
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    
    particle_faces = vtk.vtkCellArray()
    for nb in range(0, numParticle):
         # set color for each cluster
        for num_face in range(0,Particles_face[shape[nb]]):
            label = clsfct[nb]
            if shape[nb] == 0:
                Colors.InsertNextTuple3(1,0,0)
            else:
                Colors.InsertNextTuple3(color_list[label][0],color_list[label][1],color_list[label][2])
    
        i=0
        # Add the polygon to a list of polygons
        while (i<len(Particlelib[shape[nb]])):
            numVertex = Particlelib[shape[nb]][i]
            i=i+1
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(numVertex) #make a quad
            for nv in range(0, numVertex): 
                polygon.GetPointIds().SetId(nv, Particlelib[shape[nb]][i] + accumulateCornerNum[nb]) #make a face
                i =i+1
    
            particle_faces.InsertNextCell(polygon)
            
    # Next you create a vtkPolyData to store your face and vertex information that
    # represents your polyhedron.
    pd = vtk.vtkPolyData()
    pd.SetPoints(vertex_locations)
    pd.SetPolys(particle_faces)
    
    pd.GetCellData().SetScalars(Colors)
    #---------------------#
    #    visualization    #
    #---------------------#
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(bd)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetColor(1,0,0) 
    
    mapperC = vtk.vtkPolyDataMapper()
    mapperC.SetInputData(cbd)
    actorC = vtk.vtkActor()
    actorC.SetMapper(mapperC)
    actorC.GetProperty().SetColor(1,0,0) 
    
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(pd)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetColor(0.5,0.5,0.5) 
    actor2.GetProperty().SetOpacity(1) 
    
    # -------------------------Set axes ---------------------------------
    transform = vtk.vtkTransform()
    transform.Translate(2.5, 1.0, 2.5)
    axes = vtk.vtkAxesActor()
    #  The axes are positioned with a user transform
    axes.SetUserTransform(transform)
    
    # -------------------------Set Camera ---------------------------------
    camera = vtk.vtkCamera()
    cameraX, cameraY, cameraZ= 0, 0, 0
    camera.SetPosition(float(cameraX), float(cameraY), float(cameraZ))
    camera.SetFocalPoint(2.5, 0.75, 0.5)
    
    ren = vtk.vtkRenderer()
    ren.SetActiveCamera(camera);
    ren.AddActor(actor1)
    ren.AddActor(actorC)
    ren.AddActor(actor2)
    ren.SetBackground(1,1,1) # Background color white
    
    renw = vtk.vtkRenderWindow()
    renw.AddRenderer(ren)
    
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renw)

    renw.Render()

    iren.Start()
#     end = input("Terminate program? ")
#     raise SystemExit

def plot(classification,v3d,library):
    numBoundary= numParticle = numCircularBoundary =0
    labelfile = pd.read_csv(classification)
    part_num = labelfile["part_num"]
#     print(part_num)
    data, Master_list = get_data_v3d(v3d, part_num)
    clsfct,num_label = get_class(classification,Master_list)
#     print(len(clsfct))
    visualization(library,data,clsfct,num_label)

plot('labels_movement_cross_cb.csv','CenterBinding.v3d','library.vlb')





