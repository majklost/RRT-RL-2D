

class SpringParams:

    def __init__(self, stiffness, damping):
        self.stiffness = stiffness
        self.damping = damping


STANDARD_StructuralSpringParams = SpringParams(20, 20)
STANDARD_ShearSpringParams = SpringParams(3, 5)

