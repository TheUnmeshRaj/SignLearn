import mediapipe as mp

REFERENCE_IMAGES = {
    'a': 'ref_imgs/a.jpg', 'b': 'ref_imgs/b.jpg', 'c': 'ref_imgs/c.jpg',
    'd': 'ref_imgs/d.jpg', 'e': 'ref_imgs/e.jpg', 'f': 'ref_imgs/f.jpg',
    'g': 'ref_imgs/g.jpg', 'h': 'ref_imgs/h.jpg', 'i': 'ref_imgs/i.jpg',
    'j': 'ref_imgs/j.jpg', 'k': 'ref_imgs/k.jpg', 'l': 'ref_imgs/l.jpg',
    'm': 'ref_imgs/m.jpg', 'n': 'ref_imgs/n.jpg', 'o': 'ref_imgs/o.jpg',
    'p': 'ref_imgs/p.jpg', 'q': 'ref_imgs/q.jpg', 'r': 'ref_imgs/r.jpg',
    's': 'ref_imgs/s.jpg', 't': 'ref_imgs/t.jpg', 'u': 'ref_imgs/u.jpg',
    'v': 'ref_imgs/v.jpg', 'w': 'ref_imgs/w.jpg', 'x': 'ref_imgs/x.jpg',
    'y': 'ref_imgs/y.jpg', 'z': 'ref_imgs/z.jpg'
}

def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    return mp_hands, hands, mp_draw
