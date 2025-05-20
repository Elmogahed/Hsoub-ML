import gym

env = gym.make("Taxi-v3",render_mode="ansi").env

env.reset(seed=0)

env = env.unwrapped

arr=env.render()
print(arr)

print("Action Space ", env.action_space)
print("State Space ", env.observation_space)

# (taxi row, taxi column, passenger index, destination index)
state = env.encode(3, 1, 2, 0)
print("State:", state)
env.s = state
arr=env.render()
print(arr)

print(env.P[env.s])

# تحديد الحالة الابتدائية
state = env.encode(3, 1, 2, 0)
# إسناد الحالة للبيئة
env.s = state

# عداد الحركات
epochs = 0
# عداد مرات  الجزاء
# لركوب أو إنزال خاطئ
penalties = 0

# حفظ الحالات للتحريك لاحقا
frames = []  # for animation
# متغير للدلالة على الوصول للهدف
done = False
# كرر طالما لم نصل للهدف
while not done:
    # اختيار الفعل بشكل عشوائي
    action = env.action_space.sample()
    # الانتقال للحالة التالية
    state, reward, done, _, _ = env.step(action)
    # جمع عدد مرات الجزاء
    if reward == -10:
        penalties += 1

    # إضافة إطار للعرض لاحقاً
    frames.append({
        'state': state,
        'action': action,
        'reward': reward
    }
    )

    epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


from IPython.display import clear_output
# مكتبة التوقيت
from time import sleep
# دالة لمحاكة التجوال
def print_frames(frames):
    actions=['North','South','East','West','Pick-up','Drop-of']
    for i, frame in enumerate(frames):
        # الحالة التالية
        env.s = frame['state']
        # مسح الإظهار
        clear_output(wait=True)
       # الإظهار
        arr=env.render()
        print(arr)
        # اسم الفعل
        # طباعة المعلومات
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {actions[frame['action']]}")
        print(f"Reward: {frame['reward']}")
        sleep(0.1) # sleep 0.1 second
# استدعاء الدالة
print_frames(frames)

# تقييم الأداء دون التعلم

# عدد الحلقات
episodes = 100
# عداد الحركات
total_epochs = 0
# عداد العقوبات
total_penalties = 0

for _ in range(episodes):
    # حالة ابتدائية عشوائية
    state = env.reset()
    # عداد أخطاء الحلقة
    penalties = 0
    # عداد أخطاء الحلقة
    reward = 0
    # عداد الحركات
    epochs = 0
    done = False
    # الدوران حتى الوصول إلى الهدف
    while not done:

        # اختيار الفعل بشكل عشوائي
        action = env.action_space.sample()
        # الانتقال
        state, reward, done, _, _ = env.step(action)
        # عداد الأخطاء
        if reward == -10:
            penalties += 1
        # عداد الحركات
        epochs += 1
    # العدد الكلي للأخطاء
    total_penalties += penalties
    # العدد الكلي للحركات
    total_epochs += epochs
# طباعة التقييم
print(f"Results after {episodes} episodes:")
# وسطي عدد الحركات للوصول إلى الهدف
print(f"Average timesteps per episode: {(total_epochs / episodes)}")
# وسطي عدد الأخطاء
print(f"Average penalties per episode: {total_penalties / episodes}")

# المكتبة نمباي
import numpy as np
# عدد الحالات
states=env.observation_space.n
# عدد الأفعال
actions=env.action_space.n
# إنشاء مصفوفة نمباي كلها أصفار
# عدد الصفوف هو عدد الحالات
# عدد الأعمدة هو عدد الأفعال
q_table = np.zeros([states, actions])
print(q_table)
# تدريب الوكيل
# المعاملات الأساسية
alpha = 0.4
gamma = 0.6

# عدد الحلقات
episodes = 100000

for i in range(episodes):
    # إعادة التهيئة وتعيين حالة ابتدائية عشوائية
    state = env.reset()[0]
    # عداد الحركات
    epochs = 0
    # عداد الأخطاء في الركوب و التنزيل
    penalties = 0
    done = False
    # الدوران حتى الوصول للهدف
    while not done:
        action = np.argmax(q_table[state]) # القيمة العظمى لقيم التعلم
        # القيمة السابقة
        old_value = q_table[state, action]
        # الانتقال للحالة التالية
        next_state, reward, done, _,_ = env.step(action)
        next_state=next_state
        # أكبر قيمة في الحالة التالية
        next_max = np.max(q_table[next_state])
        # القيمة الجديدة
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        # تعديل القيمة
        q_table[state, action] = new_value
        # الانتقال للحالة التالية
        state = next_state
# الانتهاء من التعلم
print("Training finished.\n")
print(q_table)
# تقييم الأداء بعد التعلم
# عدد الحلقات
episodes = 100
# عداد الحركات
total_epochs = 0
# عداد العقوبات
total_penalties = 0

for _ in range(episodes):
    # حالة ابتدائية عشوائية
    state = env.reset()[0]
    # عداد أخطاء الحلقة
    penalties = 0
    # عداد أخطاء الحلقة
    reward = 0
    epochs = 0
    done = False
    # الدوران حتى الوصول إلى الهدف
    while not done:

        # اختيار الفعل الموافق لأكبر قيم تعلم للحالة الحالية
        action = np.argmax(q_table[state])
        # الانتقال
        state, reward, done, _, _ = env.step(action)
        # عداد الأخطاء
        if reward == -10:
            penalties += 1
        # عداد الحركات
        epochs += 1
    # العدد الكلي للأخطاء
    total_penalties += penalties
    # العدد الكلي للحركات
    total_epochs += epochs
# طباعة التقييم
print(f"Results after {episodes} episodes:")
# وسطي عدد الحركات للوصول إلى الهدف
print(f"Average timesteps per episode: {(total_epochs / episodes)}")
# وسطي عدد الأخطاء
print(f"Average penalties per episode: {total_penalties / episodes}")

# تحديد الحالة الابتدائية
state = env.encode(3, 1, 2, 0)
print("State:", state)
# إسناد الحالة للبيئة
env.s = state
# الإظهار
arr = env.render()
print(arr)
# حفظ الحالات للتحريك لاحقا
frames = []  # for animation
# متغير للدلالة على الوصول للهدف
done = False
# كرر طالما لم نصل للهدف
while not done:
    # اختيار الفعل الموافق لأكبر قيم تعلم للحالة الحالية
    action = np.argmax(q_table[state])
    # الانتقال للحالة التالية
    state, reward, done, _, _ = env.step(action)

    # إضافة إطار للعرض لاحقاً
    frames.append({
        'state': state,
        'action': action,
        'reward': reward
    }
    )
# استدعاء التحريك
print_frames(frames)









