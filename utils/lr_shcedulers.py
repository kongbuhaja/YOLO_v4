import math

class warmup_lr_scheduler():
    def __init__(self, warmup_max_step, init_lr):
        self.warmup_max_step = warmup_max_step
        self.init_lr = init_lr

    def __call__(self, warmup_step):
        lr = self.init_lr / self.warmup_max_step * (warmup_step + 1)
        return lr

class step_lr_scheduler():
    def __init__(self, init_lr, step_per_epoch):
        self.init_lr = init_lr
        self.step_per_epoch = step_per_epoch
    
    def __call__(self, step):
        epoch = (step + 1) / self.step_per_epoch
        if epoch < 200:
            lr = self.init_lr 
        elif epoch < 300:
            lr = self.init_lr * 0.5
        elif epoch < 400:
            lr = self.init_lr * 0.1
        elif epoch < 500:
            lr = self.init_lr * 0.05
        else:
            lr = self.init_lr * 0.01
        return lr
    
class poly_lr_scheduler():
    def __init__(self, init_lr, max_step, power=0.9):
        self.init_lr = init_lr
        self.max_step = max_step
        self.power = power
    
    def __call__(self, step):
        lr = self.init_lr * (1 - (step/(self.max_step)))**self.power
        return lr
    
class cosine_annealing_lr_scheduler():
    def __init__(self, init_lr, t_step=10, t_mult=2, min_lr=1e-6):
        self.init_lr = init_lr
        self.t_step = t_step
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
            
        lr = self.min_lr + 0.5 * (self.init_lr - self.min_lr) * (1 + math.cos(t_cur / self.t_max * math.pi))
        return lr
    
class LR_scheduler():
    def __init__(self, step_per_epoch, max_step, warmup_max_step, lr_type, init_lr):
        self.step_per_epoch = step_per_epoch
        self.max_step = max_step
        self.warmup_max_step = warmup_max_step
        self.lr_type = lr_type
        self.init_lr = init_lr
        self.warmup_lr_scheduler = warmup_lr_scheduler(self.warmup_max_step, self.init_lr)
        if self.lr_type == 'cosine_annealing':
            self.lr_scheduler = cosine_annealing_lr_scheduler(self.init_lr)
        elif self.lr_type == 'poly':
            self.lr_scheduler = poly_lr_scheduler(self.init_lr, self.max_step)
        elif self.lr_type == 'step':
            self.lr_scheduler = step_lr_scheduler(self.init_lr, self.step_per_epoch)

    def __call__(self, step, warmup_step):
        lr = self.lr_scheduler(step - self.warmup_max_step)
        if warmup_step < self.warmup_max_step:
            lr = self.warmup_lr_scheduler(warmup_step)
        return lr
