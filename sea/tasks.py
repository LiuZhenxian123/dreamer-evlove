"""One-sentence task descriptions given to the LLM. Unknown tasks fall back to their name."""

METAWORLD = {
    "assembly": "Pick up a nut (a ring with a handle) and place it over a vertical peg.",
    "basketball": "Pick up a basketball and dunk it into a hoop.",
    "bin-picking": "Pick up a puck from one bin and place it into the adjacent bin.",
    "box-close": "Pick up a box lid and place it on top of the box.",
    "button-press-topdown": "Press a button down from above.",
    "button-press-topdown-wall": "Reach over a wall and press a button down from above.",
    "button-press": "Press a button horizontally.",
    "button-press-wall": "Reach around a wall and press a button horizontally.",
    "coffee-button": "Push the button on a coffee machine.",
    "coffee-pull": "Pull a mug away from a coffee machine to a goal position.",
    "coffee-push": "Push a mug under a coffee machine.",
    "dial-turn": "Rotate a dial by 180 degrees.",
    "disassemble": "Lift a nut (a ring with a handle) off a vertical peg.",
    "door-close": "Close a hinged door.",
    "door-lock": "Lock a door by rotating its lock clockwise.",
    "door-open": "Open a hinged door by its handle.",
    "door-unlock": "Unlock a door by rotating its lock counter-clockwise.",
    "hand-insert": "Insert the gripper into a hole in the table, pushing an object down to the goal.",
    "drawer-close": "Push a drawer closed.",
    "drawer-open": "Pull a drawer open by its handle.",
    "faucet-open": "Rotate a faucet handle counter-clockwise to open it.",
    "faucet-close": "Rotate a faucet handle clockwise to close it.",
    "hammer": "Pick up a hammer and use it to drive a nail into a wall.",
    "handle-press-side": "Press a handle down, approaching from the side.",
    "handle-press": "Press a handle down.",
    "handle-pull-side": "Pull a handle up, approaching from the side.",
    "handle-pull": "Pull a handle up.",
    "lever-pull": "Pull a lever up by 90 degrees.",
    "pick-place-wall": "Pick up a puck, lift it over a wall and place it at a goal position.",
    "pick-out-of-hole": "Pick a puck up out of a hole in the table and bring it to a goal position.",
    "pick-place": "Pick up a puck and place it at a goal position in the air.",
    "plate-slide": "Slide a plate forward into a cabinet.",
    "plate-slide-side": "Slide a plate sideways into a cabinet.",
    "plate-slide-back": "Slide a plate backward out of a cabinet.",
    "plate-slide-back-side": "Slide a plate sideways out of a cabinet.",
    "peg-insert-side": "Pick up a peg and insert it sideways into a hole.",
    "peg-unplug-side": "Pull a peg sideways out of a hole.",
    "soccer": "Kick a soccer ball into a goal.",
    "stick-push": "Pick up a stick and use it to push a thermos to a goal position.",
    "stick-pull": "Pick up a stick and use it to pull a thermos by its handle to a goal position.",
    "push": "Push a puck along the table to a goal position.",
    "push-wall": "Push a puck around a wall to a goal position.",
    "push-back": "Push a puck backward along the table to a goal position.",
    "reach": "Move the gripper to a goal position.",
    "reach-wall": "Move the gripper over a wall to a goal position.",
    "shelf-place": "Pick up a puck and place it on a shelf.",
    "sweep-into": "Sweep a puck into a hole in the table.",
    "sweep": "Sweep a puck off the table.",
    "window-open": "Slide a window open.",
    "window-close": "Slide a window closed.",
}

DMC = {
    "walker_walk": "A planar bipedal walker must stay upright and walk forward at 1 m/s.",
    "walker_run": "A planar bipedal walker must stay upright and run forward as fast as possible (8 m/s).",
    "walker_stand": "A planar bipedal walker must stand upright with its torso high.",
    "hopper_hop": "A planar one-legged hopper must hop forward while keeping its torso high.",
    "hopper_stand": "A planar one-legged hopper must balance upright with its torso high.",
    "cheetah_run": "A planar cheetah must run forward as fast as possible.",
    "quadruped_walk": "A four-legged robot must stay upright and walk forward at 0.5 m/s.",
    "quadruped_run": "A four-legged robot must stay upright and run forward at 5 m/s.",
    "quadruped_escape": "A four-legged robot on bumpy terrain must move as far from the origin as possible.",
    "cartpole_swingup": "Swing a pole up by moving the cart and balance it upright near the center.",
    "finger_spin": "A planar finger must keep a hinged body spinning.",
    "reacher_easy": "A two-link planar arm must bring its fingertip to a target.",
    "cup_catch": "Move a cup to swing a ball on a string up and catch it in the cup.",
}

SUITE_NOTES = {
    "metaworld": (
        "Sawyer robot arm with a parallel gripper on a table, seen from a fixed corner camera. "
        "The action is a 3D end-effector displacement plus a gripper open/close command. "
        "z is up; the table surface is near z = 0."
    ),
    "dmc": (
        "DeepMind Control Suite (MuJoCo), seen from a fixed tracking camera. Actions are joint torques. "
        "z is up; for planar bodies motion is along x."
    ),
}


def describe(task):
    """`task` is the env task string, e.g. 'metaworld_pick-place' or 'dmc_walker_walk'."""
    suite, name = task.split("_", 1)
    table = {"metaworld": METAWORLD, "dmc": DMC}.get(suite, {})
    text = table.get(name, f"The task is called '{name}'.")
    return f"{text} Setting: {SUITE_NOTES.get(suite, suite)}"
