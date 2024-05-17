import math
import numpy as np

class warmup():
    def __init__(self, init_lr, warmup_max_step):
        self.init_lr = init_lr
        self.warmup_max_step = warmup_max_step

    def __call__(self, step):
        lr = self.init_lr / self.warmup_max_step * step
        return lr

class step():
    def __init__(self, init_lr, steps, decays, step_per_epoch):
        self.init_lr = init_lr
        self.count = len(steps)
        self.steps = np.array(steps) * step_per_epoch
        self.decays = [np.prod(decays[:i+1]) for i in range(self.count)]
        self.step_per_epoch = step_per_epoch
    
    def __call__(self, step):
        for i in range(self.count):
            if step <= self.steps[i]:
                return self.init_lr * self.decays[i]
        return self.init_lr * self.decays[-1]
    
class poly():
    def __init__(self, init_lr, max_step, power=0.9):
        self.init_lr = init_lr
        self.max_step = max_step
        self.power = power
    
    def __call__(self, step):
        lr = self.init_lr * (1 - (step/(self.max_step)))**self.power
        return lr
    
class cosine_annealing_warm_restart():
    def __init__(self, init_lr, t_step=10, t_mult=2, min_lr=1e-6):
        self.max_lr = init_lr
        self.t_max = t_step
        self.t_mult = t_mult
        self.csum = 0
        self.min_lr = min_lr

    def __call__(self, step):
        t_cur = step - self.csum
        while(t_cur >= self.t_max):
            t_cur -= self.t_max
            self.csum += self.t_max
            self.t_max *= self.t_mult
            
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(t_cur / self.t_max * math.pi))
        return lr
    
class cosine_annealing():
    def __init__(self, init_lr, t_step=50, min_lr=1e-6):
        self.max_lr = init_lr
        self.t_max = t_step
        self.min_lr = min_lr

    def __call__(self, step):
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(step / self.t_max * math.pi))
        return lr
    
class custom():
    def __init__(self, init_lr, t_step=50, t_mult=2, min_lr=1e-6):
        self.max_lr = init_lr
        self.t_max = t_step
        self.t_mult = t_mult
        self.min_lr = min_lr
        self.cycle = (2 * t_mult-1)

    def __call__(self, step):
        t_cur = step - self.t_max
        if t_cur == self.t_max*(self.cycle-1):
            self.max_lr *= 0.97

        while(t_cur >= self.t_max*self.cycle):
            t_cur -= self.t_max
            self.t_max *= self.t_mult
            
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(step / self.t_max * math.pi))
        return lr
    
class LR_scheduler():
    def __init__(self, lr_scheduler, epochs, step_per_epoch):
        self.init_lr = lr_scheduler['lr']
        self.step_per_epoch = step_per_epoch
        self.max_step = epochs * step_per_epoch
        self.warmup_max_step = lr_scheduler['warmup_epochs'] * step_per_epoch
        self.warmup_step = 1

        self.warmup_lr_scheduler = warmup(self.init_lr, self.warmup_max_step)
        if lr_scheduler['name'] == 'cosine_annealing':
            self.lr_scheduler = cosine_annealing(self.init_lr)

        elif lr_scheduler['name'] == 'cosine_annealing_warm_restart':
            self.lr_scheduler = cosine_annealing_warm_restart(self.init_lr)

        elif lr_scheduler['name'] == 'poly':
            self.lr_scheduler = poly(self.init_lr, 
                                                  self.max_step,
                                                  lr_scheduler['power'])
            
        elif lr_scheduler['name'] == 'step':
            self.lr_scheduler = step(self.init_lr, 
                                                  lr_scheduler['steps'], 
                                                  lr_scheduler['decays'], 
                                                  self.step_per_epoch)
            
        elif lr_scheduler['name'] == 'custom':
            self.lr_scheduler = custom(self.init_lr)

    def __call__(self, step):
        if self.warmup_step <= self.warmup_max_step:
            lr = self.warmup_lr_scheduler(self.warmup_step)
            self.warmup_step += 1
        else:
            lr = self.lr_scheduler(step - self.warmup_max_step)

        return lr
