import rpyc
import math
from scipy import linalg
import time
import sys
import threading
import pytorch3d


import rpyc
import threading

import time
import os
import glob
import sys
import json
import vtk
import numpy as np



import platform

import torch
from monai.networks.nets import UNETR


import torch.nn as nn
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import Materials
from typing import Optional
from pytorch3d.renderer.blending import (hard_rgb_blend,BlendParams)
from pytorch3d.renderer.mesh.rasterizer import (Fragments)
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.lighting import PointLights
# from pytorch3d.common.types import Device

from vtk.util.numpy_support import vtk_to_numpy
from monai.networks.nets import UNet
from monai.data import decollate_batch
from monai.transforms import (AsDiscrete,ToTensor)
from scipy import linalg
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,look_at_rotation,TexturesVertex,blending

)

    

class Math :
    def __init__(self,x):
        self.x=x

    def carre(self):
        return self.x*self.x


class MyService(rpyc.Service):
    _server=None
    
    def set_server(self, server):
        self._server = server
    
        
    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass
    
    def exposed_is_ready(self):
        return True 

        
    def exposed_running(self,x):
        m = Math(x)
        return m.carre()
    
    def exposed_stop(self):
        # Informer le client de la déconnexion imminente
        threading.Thread(target=self.shutdown_server).start()
        return "DISCONNECTING"

    def shutdown_server(self):
        # Attendez quelques secondes pour donner au client le temps de se préparer
        time.sleep(2)
        self._server.close()

# Associer la fonction de fermeture du serveur au signal Ctrl+C (SIGINT)



if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    my_service_instance = MyService()  # Créez une instance de votre service
    t = ThreadedServer(my_service_instance, port=18817)
    
    my_service_instance.set_server(t)  # Passez la référence du serveur à votre instance
    
    t.logger.quiet = False
    t.logger.debug("Serveur RPyC en cours d'exécution sur le port 18812")
    t.start()

    # Attendre que le serveur soit prêt
    # while not t._threads[0].service._is_ready():
    #     time.sleep(1)

    # Récupérer la liste des connexions actives
    # active_connections = t._threads[0].service.get_active_connections()

    # # Afficher les informations sur les connexions actives
    # if active_connections:
    #     print("Connexions actives :")
    #     for conn in active_connections:
    #         print(f"   - {conn}")
    # else:
    #     print("Aucune connexion active pour le moment.")
