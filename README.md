# Preface 
Obtaining dynamic skills for autonomous machines has been a cardinal challenge in robotics. In the
field of legged systems, many attempts have been made to attain diverse skills using conventional
inverse kinematics techniques. In recent years, learning-based quadrupedal locomotion has
been achieved by reinforcement learning (RL) approaches to address more complex environments
and improve performance. However, the demand for acquiring more highly dynamic
motions has brought new challenges to robot learning. A primary shortage of motivating desired
behaviors by reward engineering is the arduous reward-shaping process involved. It can sometimes
become extremely demanding in developing highly dynamic skills such as jumping and backflipping,
where various terms of motivation and regularization require elaborated refinement.
 
<p align = "center">
<img src = "https://github.com/AYUSH-ISHAN/Tiger_code/blob/main/assets/minitaur.jpeg" align="center"/><br>
<em>Minitaur in real life</em>
</p>

# Algorithm

<p align = "center">
<img src = "./assets/latest_algo.jpg" align="center", height="600" width="650"/><br>
<em>Our algorithm</em>
</p>

# Results
<table align = "center">
<tr>
 <td>Mean Reward Vs Epochs</td>
 <td><img src = "assets/reward.png" width = "400" height = "400"/></td>
</tr>
<tr>
 <td>Mean Value Loss Vs Epochs</td>
 <td><img src = "assets/value_loss.png" width = "400" height = "400"/></td>
</tr>
<tr>
 <td>Mean Surrogate Loss Vs Epochs</td>
 <td><img src = "assets/surrogate_loss.png" width = "400" height = "400"/></td>
</tr>
 
</table>

# References

1)<a href="https://drive.google.com/file/d/1MK54cn8JUzRRfdzMHomhmVs9A3J_teJl/view?usp=sharing/">Legged Locomotion in Challenging Terrains using Egocentric Vision: Ananye Agarwal, Ashish Kumar, Jitendra Malik, Deepak Pathak</a>

2)<a href="https://drive.google.com/file/d/1dCwjm0I-G4eemxMy2tScEs0h4obKsisd/view?usp=sharing">Learning Agile Skills via Adversarial Imitation of Rough Partial 
   Demonstrations: Chenhao Li, Marin Vlastelica, Sebastian Blaes, Jonas Frey,Felix Grimminger, Georg Martius<a>
   
3)<a href="https://drive.google.com/file/d/1qdk-Ph3SiBDkevclh51l-SUrrh0ICsIi/view?usp=sharing">Deep Whole-Body Control: Learning a Unified Policy for Manipulation
and Locomotion: Zipeng Fu, Xuxin Cheng, Deepak Pathak<a>

