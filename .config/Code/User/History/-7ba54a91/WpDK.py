from threading import Thread
from time import sleep

class Foo:
    def __init__(self):
        self.somevar: int = 0
        
    def update_somevar(self):
        self.somevar += 1
        
    def start(self):
        while True:
            print("Before:", self.somevar)
            self.update_somevar()
            print("After", self.somevar)
            sleep(1)
        
# watch = Watch()
# from time import sleep
# sleep(1)
foo = Foo()
watch_thread = Thread(target=foo.start)
watch_thread.start()