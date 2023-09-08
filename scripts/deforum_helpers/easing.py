# With  0 interp_alpha, uses linear easing (no easing)
# Above 0 interp_alpha, interpolates linear with ease_in_out_quint, where +1 is full easing
# Below 0 interp_alpha, interpolates linear with ease_out_in_quint, where -1 is full easing
# 0 to -1 goes towards fast in/slow middle/fast out
# 0 is normal in/out
# 0 to 1 goes towards slow in/fast middle/slow out
def easing_linear_interp(time, interp_alpha):
    # all of time is from 0 to 1
    time = max(0, min(1, time))

    # function to mix with another
    a1 = ease_in_out_linear(time, b=0, c=1, d=1)

    if interp_alpha == 0: # 0 means linear (no interpolation)
        result = a1
    else:
        if interp_alpha > 0:
            a2 = ease_in_out_quint(time, b=0, c=1, d=1)      
        else:
            # make interp_alpha a positive number if we get to this case 
            interp_alpha = abs(interp_alpha)
            a2 = ease_out_in_quint(time, b=0, c=1, d=1)
        result = (a1 * (1 - interp_alpha)) + (a2 * interp_alpha)

    return result

# easing functions: t is time, b is begin, c is change in value, d is duration
# in_out and out_in easing functions use the helper functions in and out

def ease_in_out_linear(t, b, c, d):
    return c * t / d + b

def ease_in_out_quint(t, b, c, d):
    if t < d / 2:
        return ease_in_quint(t * 2, b, c / 2, d)
    return ease_out_quint((t * 2) - d, b + c / 2, c / 2, d)

def ease_out_in_quint(t, b, c, d):
    if t < d / 2:
        return ease_out_quint(t * 2, b, c / 2, d)
    return ease_in_quint((t * 2) - d, b + c / 2, c / 2, d)

def ease_in_quint(t, b, c, d):
    t /= d
    return c * t ** 5 + b

def ease_out_quint(t, b, c, d):
    t /= d
    t -= 1
    return c * (t ** 5 + 1) + b


# Extra functions

# def easing_functions():
#   return {
#     'in_out_linear': ease_in_out_linear,
#     'in_out_quad': ease_in_out_quad,
#     'in_out_cubic': ease_in_out_cubic,
#     'in_out_quart': ease_in_out_quart,
#     'in_out_quint': ease_in_out_quint,
#     'in_out_sine': ease_in_out_sine,
#     'out_in_quad': ease_out_in_quad,
#     'out_in_cubic': ease_out_in_cubic,
#     'out_in_quart': ease_out_in_quart,
#     'out_in_quint': ease_out_in_quint,
#     'out_in_sine': ease_out_in_sine,
#     'in_quad': ease_in_quad,
#     'in_cubic': ease_in_cubic,
#     'in_quart': ease_in_quart,
#     'in_quint': ease_in_quint,
#     'in_sine': ease_in_sine,
#     'out_quad': ease_out_quad,
#     'out_cubic': ease_out_cubic,
#     'out_quart': ease_out_quart,
#     'out_quint': ease_out_quint,
#     'out_sine': ease_out_sine
#   }

# def ease_in_out_linear(t, b, c, d):
#     return c * t / d + b

# def ease_in_out_quint(t, b, c, d):
#     if t < d / 2:
#         return ease_in_quint(t * 2, b, c / 2, d)
#     return ease_out_quint((t * 2) - d, b + c / 2, c / 2, d)

# def ease_in_out_quad(t, b, c, d):
#     if t < d / 2:
#         return ease_in_quad(t * 2, b, c / 2, d)
#     return ease_out_quad((t * 2) - d, b + c / 2, c / 2, d)

# def ease_in_out_cubic(t, b, c, d):
#     if t < d / 2:
#         return ease_in_cubic(t * 2, b, c / 2, d)
#     return ease_out_cubic((t * 2) - d, b + c / 2, c / 2, d)

# def ease_in_out_quart(t, b, c, d):
#     if t < d / 2:
#         return ease_in_quart(t * 2, b, c / 2, d)
#     return ease_out_quart((t * 2) - d, b + c / 2, c / 2, d)

# def ease_in_out_sine(t, b, c, d):
#     return -c / 2 * (math.cos(math.pi * t / d) - 1) + b

# def ease_out_in_linear(t, b, c, d):
#     return -c * t / d + c + b

# def ease_out_in_quint(t, b, c, d):
#     if t < d / 2:
#         return ease_out_quint(t * 2, b, c / 2, d)
#     return ease_in_quint((t * 2) - d, b + c / 2, c / 2, d)

# def ease_out_in_quad(t, b, c, d):
#     if t < d / 2:
#         return ease_out_quad(t * 2, b, c / 2, d)
#     return ease_in_quad((t * 2) - d, b + c / 2, c / 2, d)

# def ease_out_in_cubic(t, b, c, d):
#     if t < d / 2:
#         return ease_out_cubic(t * 2, b, c / 2, d)
#     return ease_in_cubic((t * 2) - d, b + c / 2, c / 2, d)

# def ease_out_in_quart(t, b, c, d):
#     if t < d / 2:
#         return ease_out_quart(t * 2, b, c / 2, d)
#     return ease_in_quart((t * 2) - d, b + c / 2, c / 2, d)

# def ease_out_in_sine(t, b, c, d):
#     if t < d / 2:
#         return ease_out_sine(t * 2, b, c / 2, d)
#     return ease_in_sine((t * 2) - d, b + c / 2, c / 2, d)

# # helper easing functions

# def ease_out_quint(t, b, c, d):
#     t /= d
#     t -= 1
#     return c * (t ** 5 + 1) + b

# def ease_out_quint(t, b, c, d):
#     t /= d
#     t -= 1
#     return c * (t ** 5 + 1) + b

# def ease_out_quad(t, b, c, d):
#     t /= d
#     return -c * t * (t - 2) + b

# def ease_out_cubic(t, b, c, d):
#     t /= d
#     t -= 1
#     return c * (t ** 3 + 1) + b

# def ease_out_quart(t, b, c, d):
#     t /= d
#     t -= 1
#     return -c * (t ** 4 - 1) + b

# def ease_out_sine(t, b, c, d):
#     return c * math.sin(t / d * (math.pi / 2)) + b

# def ease_in_quint(t, b, c, d):
#     t /= d
#     return c * t ** 5 + b

# def ease_in_quad(t, b, c, d):
#     t /= d
#     return c * t * t + b

# def ease_in_cubic(t, b, c, d):
#     t /= d
#     return c * t ** 3 + b

# def ease_in_quart(t, b, c, d):
#     t /= d
#     return c * t ** 4 + b

# def ease_in_sine(t, b, c, d):
#     return -c * math.cos(t / d * (math.pi / 2)) + c + b

