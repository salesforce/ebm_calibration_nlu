import math
import numpy as np
import random
import torch

class Scheduler(object):
    def __init__(self, base, is_top_p=False):
        self.last_step_num = 0
        self.last_step_value = 0
        self.is_top_p = is_top_p 
        if is_top_p:
            self.base = float(base)
        else:
            self.base = int(base)

    def step(self):
        raise NotImplementedError("Step has not been implemented in the abstract class.")

class RandomSpaceTopkScheduler(Scheduler):
    def __init__(self, base, randomspace_rate, preserve_largest_prob, is_top_p=False):
        """
        Implements  
        """
        if is_top_p == True:
            raise NotImplementedError("only topk for now.")
        super().__init__(base, is_top_p=is_top_p)
        self.random_rate = float(randomspace_rate)
        self.preserve_largest_prob = int(preserve_largest_prob)
        assert(self.preserve_largest_prob == 0 or self.preserve_largest_prob == 1)

    def transform(self, logits):
        top_k = self.base
        tt = torch.sort(logits, descending = True)
        val, ind = tt[0][:, :top_k], tt[1][:, :top_k]
        assert(val.size(1) == top_k)
        #logits.fill_(least_value)
        
        least_value = torch.min(val).item() - 1000
        logits.fill_(least_value)
        
        ra = torch.ones(val.size()).cuda().uniform_()
        if self.preserve_largest_prob == 1:
            ra[:, 0] = 2 #we never mask out the first token, so that at least one token will survive
        
        bo = ra < self.random_rate
        if self.preserve_largest_prob == 0:
            for i in range(bo.size(0)):
                if (bo[i]).all().item() == True: #all number will be least_value, prevent it
                    #print('debug', i, 'all True')
                    bo[i][random.randint(0, bo.size(1) - 1)] = False
        val[bo] = least_value

        logits = logits.scatter(1, ind, val)
        return logits

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base 

"""
sche = RandomSpaceTopkScheduler(5, 0.95, 0)
a = torch.log(torch.softmax(torch.randn(10, 6).cuda(), dim = -1))
print(a)
a = sche.transform(a)
print(a)
"""

class SortedNoisedFixedScheduler(Scheduler):
    def __init__(self, noise_weight, base, is_top_p = False):
        """
        Implements  
        """
        self.noise_weight = float(noise_weight) #ignore the other base
        assert(self.noise_weight < 1.0)
        #print('NoisedTemperatureScheduler noise_scale:', self.noise_scale)
        assert(is_top_p == False) #only do topk for now
        super().__init__(base, is_top_p=is_top_p)
    
    def transform(self, logits):
        #the temperature transform is done outside
        #we only do topk for now
        topk_logits, topk_indices = torch.sort(logits, descending = True)
        topk_logits, topk_indices = topk_logits[:, :self.base], topk_indices[:, :self.base]
        probs = torch.softmax(topk_logits, dim = -1) 

        unif = torch.zeros(logits.size(0), self.base).uniform_().cuda()
        unif[unif < 1e-09] = 1e-09
        unif[unif > 1 - 1e-09] = 1 - 1e-09
        log_unif = - torch.log(unif)
        unif_simplex = log_unif / torch.sum(log_unif, dim = -1).view(-1, 1)
        sort_s = torch.sort(unif_simplex, dim = -1, descending = True)[0] #the only different

        #noise = torch.randn(probs.size()).cuda().uniform_()
        probs = probs * (1 - self.noise_weight) + self.noise_weight * sort_s
        probs = probs / probs.sum(dim = -1).view(-1, 1)
        topk_logits = torch.log(probs)
        
        least_value = logits.min().item() - 1000
        logits.fill_(least_value)
        logits = logits.scatter(1, topk_indices, topk_logits)

        return logits.detach()

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base 

class NoisedTemperatureScheduler(Scheduler):
    def __init__(self, noise_weight):
        """
        Implements  
        """
        self.noise_weight = float(noise_weight) #ignore the other base
        assert(self.noise_weight < 1.0)
        #print('NoisedTemperatureScheduler noise_scale:', self.noise_scale)
        super().__init__(1, is_top_p=False)
        self.base = None
    
    def transform(self, logits):
        #the temperature transform is done outside
        indices_des = torch.sort(logits, descending = True)[1][:, :]
        probs = torch.softmax(logits, dim = -1) 

        unif = torch.zeros(logits.size(0), logits.size(1)).uniform_().cuda()
        unif[unif < 1e-09] = 1e-09
        unif[unif > 1 - 1e-09] = 1 - 1e-09
        log_unif = - torch.log(unif)
        unif_simplex = log_unif / torch.sum(log_unif, dim = -1).view(-1, 1)

        #noise = torch.randn(probs.size()).cuda().uniform_()
        probs = probs * (1 - self.noise_weight) + self.noise_weight * unif_simplex
        probs = probs / probs.sum(dim = -1).view(-1, 1)
        logits = torch.log(probs)
        return logits.detach()

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base 

class TargetEntropyScheduler(Scheduler):
    def __init__(self, base, is_top_p=False):
        """
        Implements  
        """
        if is_top_p == True:
            raise NotImplementedError("only topk for now.")
        self.target_entropy = float(base) #ignore the other base
        super().__init__(1, is_top_p=is_top_p)
        self.base = None
    
    def compute_entropy(self, logits):
        distro = torch.softmax(logits, dim = -1)
        entropy = - torch.sum(distro * torch.log(distro + 1e-10), dim = - 1)
        return entropy

    def transform(self, logits):
        e_tar = torch.zeros(logits.size(0)).cuda()
        scale = torch.ones(logits.size(0)).cuda()
        e_tar.fill_(self.target_entropy)
        e_cur = self.compute_entropy(logits).squeeze()
        for kk in range(30):
            scale.fill_(1)
            ss = 1.2 if kk < 10 else 1.02
            if kk > 20: ss = 1.002
            scale[e_cur < e_tar] = 1 / ss
            scale[e_cur > e_tar] = ss
            logits = (logits * scale.unsqueeze(1)).detach()
            e_cur = self.compute_entropy(logits).squeeze()
        if random.random() < 0.001: 
            print('random debug target entropy gap:', (e_cur - e_tar).abs().mean()) 
        return logits.detach()

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base 

class MaxEntropyScheduler(Scheduler):
    def __init__(self, base, is_top_p=False):
        """
        Implements  
        """
        if is_top_p == True:
            raise NotImplementedError("only topk for now.")
        self.max_entropy = float(base) #ignore the other base
        super().__init__(1, is_top_p=is_top_p)
        self.base = None
    
    def compute_entropy(self, logits):
        distro = torch.softmax(logits, dim = -1)
        entropy = - torch.sum(distro * torch.log(distro + 1e-10), dim = - 1)
        return entropy

    def transform(self, logits):
        e_max = torch.zeros(logits.size(0)).cuda()
        scale = torch.ones(logits.size(0)).cuda()
        e_max.fill_(self.max_entropy)
        e_cur = self.compute_entropy(logits).squeeze()
        for kk in range(30):
            scale.fill_(1)
            ss = 1.3
            if kk > 20: ss = 1.1
            scale[e_cur > e_max * ss] = ss
            logits = (logits * scale.unsqueeze(1)).detach()
            e_cur = self.compute_entropy(logits).squeeze()
        if random.random() < 0.005: 
            print('random debug target entropy gap:', ((e_cur - e_max).abs() * (e_cur > e_max).float()).max()) 
        return logits.detach()

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base 

"""
sche = TargetEntropyScheduler(2.5)
a = torch.randn(4, 20).cuda()
print(a)
sche.transform(a)
"""
class UniformSimplexScheduler(Scheduler):
    def __init__(self, base, is_top_p=False):
        """
        Implements  
        """
        if is_top_p == True:
            raise NotImplementedError("only topk for now.")
        super().__init__(base, is_top_p=is_top_p)
    
    def transform(self, logits, least_value, temperature = 1.0):
        top_k = self.base
        indices_keep = torch.sort(logits, descending = True)[1][:, :top_k]
        
        unif = torch.zeros(logits.size(0), top_k).uniform_().cuda()
        unif[unif < 1e-09] = 1e-09
        unif[unif > 1 - 1e-09] = 1 - 1e-09

        log_unif = - torch.log(unif)
        log_unif_simplex = torch.log(log_unif / torch.sum(log_unif, dim = -1).view(-1, 1))
        
        if temperature != 1.0:
            log_unif_simplex = log_unif_simplex / temperature
        
        least_value = min(torch.min(log_unif_simplex).item() - 1000, least_value)
        logits.fill_(least_value)
        if torch.sum(torch.isinf(log_unif_simplex)) > 0:
            print('error! meet inf')
            sys.exit(1)
        #sorted_simplex, _ = torch.sort(unif_simplex, dim = -1, descending = True)
        #least_value = min(least_value, torch.min(log_unif_simplex).item() - 1000)
        #logits.fill_(least_value)
        """
        if torch.min(log_unif_simplex).item() - 900 <= least_value:
            print('min!!', torch.min(log_unif_simplex).item(), least_value)
            breakpoint()
            sys.exit(1)
        """
        logits = logits.scatter(1, indices_keep, log_unif_simplex)
        return logits

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base 

class SortedSimplexScheduler(Scheduler):
    def __init__(self, base, is_top_p=False):
        """
        Implements  
        """
        if is_top_p == True:
            raise NotImplementedError("only topk for now.")
        super().__init__(base, is_top_p=is_top_p)
    
    def transform(self, logits, least_value, temperature = 1.0):
        top_k = self.base
        indices_keep = torch.sort(logits, descending = True)[1][:, :top_k]
        #logits.fill_(least_value)
        
        unif = torch.zeros(logits.size(0), top_k).uniform_().cuda()
        unif[unif < 1e-09] = 1e-09
        unif[unif > 1 - 1e-09] = 1 - 1e-09

        log_unif = - torch.log(unif)
        log_unif_simplex = torch.log(log_unif / torch.sum(log_unif, dim = -1).view(-1, 1))
        sorted_s = torch.sort(log_unif_simplex, dim = -1, descending = True)[0] #the only different
 
        if temperature != 1.0:
            sorted_s = sorted_s / temperature
        
        least_value = min(torch.min(sorted_s).item() - 1000, least_value)
        logits.fill_(least_value)
        
        if torch.sum(torch.isinf(log_unif_simplex)) > 0:
            print('error! meet inf!')
            sys.exit(1)

        #sorted_simplex, _ = torch.sort(unif_simplex, dim = -1, descending = True)
        #least_value = min(least_value, torch.min(log_unif_simplex).item() - 1000)
        #logits.fill_(least_value)
        """
        if torch.min(log_unif_simplex).item() - 900 <= least_value:
            print('min!!', torch.min(log_unif_simplex).item(), least_value)
            breakpoint()
            sys.exit(1)
        """
        logits = logits.scatter(1, indices_keep, sorted_s)
        return logits

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base 

"""
sche = SortedSimplexScheduler(4)
a = torch.randn(3, 6).cuda()
print(a)
sche.transform(a, -100)
"""

class FixedScheduler(Scheduler):
    def __init__(self, base, is_top_p=False):
        """
        Implements a fixed sampling scheduler, which is always constant. 
        """
        super().__init__(base, is_top_p=is_top_p)
    
    def __str__(self):
        if self.is_top_p == False:
            return 'TOPK' + str(self.base)
        else:
            return 'TOPP' + str(self.base) 

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base 
    
    def transform(self, logits, least_value = -1000):
        assert(self.is_top_p == False)
        top_k = self.base
        vals_keep, indices_keep = torch.sort(logits, descending = True)
        vals_keep, indices_keep = vals_keep[:, :top_k], indices_keep[:, :top_k]
        #logits.fill_(least_value)
        
        new_logits = torch.ones(logits.size()).cuda()
        new_logits.fill_(least_value)
        
        if torch.sum(torch.isinf(new_logits)) > 0:
            print('error! meet inf!')
            sys.exit(1)

        #sorted_simplex, _ = torch.sort(unif_simplex, dim = -1, descending = True)
        #least_value = min(least_value, torch.min(log_unif_simplex).item() - 1000)
        #logits.fill_(least_value)
        """
        if torch.min(log_unif_simplex).item() - 900 <= least_value:
            print('min!!', torch.min(log_unif_simplex).item(), least_value)
            breakpoint()
            sys.exit(1)
        """
        new_logits = new_logits.scatter(1, indices_keep, vals_keep)
        return new_logits

class ExpScheduler(Scheduler):
    def __init__(self, base, is_top_p=False):
        """
        Implements a sampling scheduler following base^x, where base is an integer.

        """
        super().__init__(base, is_top_p=is_top_p)
        
    def step(self):
        self.last_step_num += 1
        return self.base**self.last_step_num

class FibScheduler(Scheduler):
    def __init__(self, base=3, is_top_p=False):
        """
        Implements a sampling scheduler following base^x, where base is an integer.

        """
        super().__init__(base, is_top_p=is_top_p)
       
        self.second_step_value = 1
        self.last_step_value = 2
        self.phi = (1 + math.sqrt(5)) / 2

    def step(self):
        self.last_step_num += 1
        if self.last_step_num==1:
            return 2
        fib = round(pow(self.phi, self.base)/math.sqrt(5))
        self.base += 1
        return min(fib, 100)

class LinearScheduler(Scheduler):
    def __init__(self, base, is_top_p=False):
        """
        Implements a sampling scheduler that progressively increases from base. 

        """
        super().__init__(base, is_top_p=is_top_p)
        
    def step(self):
        self.last_step_num += 1
        self.base += 1
        return self.base

class LinearLagScheduler(Scheduler):
    def __init__(self, base=2, lag=2, is_top_p=False):
        """
        Implements a sampling scheduler that progresses every LAG iterations from BASE. 

        """
        super().__init__(base, is_top_p=is_top_p)
        self.lag = int(lag)

    def step(self):
        self.last_step_num += 1
        if (self.last_step_num % self.lag)==0:
            self.base += 1
        return self.base


class NegLinearScheduler(Scheduler):
    def __init__(self, base, is_top_p=False):
        """
        Implements a sampling scheduler that progressively decreases the scheduler. 
        """

        super().__init__(base, is_top_p=is_top_p)

    def step(self):
        self.last_step_num += 1
        self.base -= 1
        return max(self.base, 2)

class SineScheduler(Scheduler):
    def __init__(self, base, period=10, amplitude=10, is_top_p=False):
        """
        Implements a sampling scheduler following sin(x). 
        Where we start at a base value and progressively increase or decrease it. 
        """
        super().__init__(base, is_top_p=is_top_p)

        if is_top_p:
            self.amp = float(amplitude)
            self.period = float(period)
        else:
            self.base = int(base)
            self.amp = int(amplitude)
            self.period = int(period)

        self.sin_period = (2*math.pi) / self.period
        self.is_top_p = is_top_p

    def step(self):
        self.last_step_num += 1
        if self.is_top_p and self.last_step_num==1:
            return 0.95
        # TODO: another sweep with this enabled!
        # if (self.last_step_num % self.period) in [self.period//2, 0]:
            # self.amp = math.ceil(self.last_step_num / self.period) * self.amp * 1.5

        additional = math.sin(self.sin_period * self.last_step_num)*self.amp
        if self.is_top_p:
            return max(min(float(self.base + abs(additional)), 1.0), 0.10)
        else:
            return max(int(self.base + abs(additional)), 1)

class TanhScheduler(Scheduler):
    def __init__(self, base=15, window=60, is_top_p=False):
        """
        Implements a sampling scheduler following tanh(x). 
        Where we start at a base value and progressively increase it. 
        """
        super().__init__(base, is_top_p=is_top_p)
        self.window = int(window) 
        self.last_step_num = 8

    def step(self):
        self.last_step_num += 1
        return int(max(np.tanh(self.last_step_num / self.window) * self.base, 1)) + 1 

class SineExpScheduler(Scheduler):
    def __init__(self, base, period=10, amplitude=10, is_top_p=False):
        """
        Implements a sampling scheduler following sin(x). 
        Where we start at a base value and progressively increase or decrease it. 
        """
        super().__init__(base, is_top_p=is_top_p)

        if is_top_p:
            self.amp = float(amplitude)
            self.period = float(period)
        else:
            self.base = int(base)
            self.amp = int(amplitude)
            self.period = int(period)

        self.sin_period = (2*math.pi) / self.period
        self.is_top_p = is_top_p

    def step(self):
        self.last_step_num += 1
        if self.is_top_p and self.last_step_num==1:
            return 0.95
        # TODO: another sweep with this enabled!
        # if (self.last_step_num % self.period) in [self.period//2, 0]:
            # self.amp = math.ceil(self.last_step_num / self.period) * self.amp * 1.5

        additional = math.sin(self.sin_period * self.last_step_num) * (2**self.amp)
        if self.is_top_p:
            return max(min(float(self.base + abs(additional)), 1.0), 0.10)
        else:
            return max(int(self.base + abs(additional)), 1)

class JointScheduler(Scheduler):
    def __init__(self, top_k, top_p, is_top_p=False):
        super().__init__(-1, is_top_p=is_top_p)
        self.top_k = int(top_k)
        self.top_p = float(top_p)

    def step(self):
        self.last_step_num += 1
        return {"top_k": self.top_k, "top_p": self.top_p} 

class RandomFixedScheduler(FixedScheduler):
    def __init__(self, base, max_randomness, is_top_p=False):
        super().__init__(base, is_top_p=is_top_p)
        if is_top_p:
            self.max_randomness = float(max_randomness)
        else:
            self.max_randomness = int(max_randomness)

    def step(self):
        if self.is_top_p:
            randomness = random.uniform(0.01, self.max_randomness)
        else:
            randomness = random.randint(1, self.max_randomness)

        return self.base + randomness 

class TemperatureSweep(Scheduler):
    """ 
    An empty class that is fixed to is_top_p=True and p=1.0
    Useful for checking the results of a temperature sweep for quality diversity tradeoff.
    """
    def __init__(self, base, is_top_p=False):
        super().__init__(base, is_top_p=is_top_p)

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base 

class NegativeScheduler(Scheduler):
    """
    Instead of doing Top K sampling, we are going to do Top -K (negative-K) sampling.
    That is, remove the top K logits, and then sample.
    This is to verify our approach for generation.
    """
    def __init__(self, base, negative_base, is_top_p=False):
        super().__init__(base, is_top_p=is_top_p)
        if is_top_p:
            self.negative_base = float(negative_base)
        else:
            self.negative_base = int(negative_base)

    def step(self):
        self.last_step_num += 1.0
        self.last_step_value = self.base
        return self.base, self.negative_base 
