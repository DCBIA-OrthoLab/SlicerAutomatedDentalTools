import rpyc
import math


class MyService(rpyc.Service):
    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_add_function(self, func_name, func_code):
        exec(func_code, globals())
        if not func_name == "imports":  # Ne pas ajouter "imports" comme une fonction
            setattr(self, f'exposed_{func_name}', eval(func_name))

    def exposed_exec_code(self, code):
        try:
            exec(code)
            return True
        except Exception as e:
            return str(e)


if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MyService, port=18812)
    t.start()
