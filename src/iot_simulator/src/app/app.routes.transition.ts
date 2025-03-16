import {
    animate,
    query,
    style,
    transition,
    trigger,
} from "@angular/animations";

export const floatAnimation = trigger("floatAnimation", [
    transition("_ => *", [
        query(
            ":enter",
            [
                style({
                    transform: "translateX(100%) translateY(-50%)",
                    opacity: 0,
                }),
                animate(
                    "0.5s ease-in-out",
                    style({
                        transform: "translateX(0) translateY(-50%)",
                        opacity: 1,
                    }),
                ),
            ],
            { optional: true },
        ),
    ]),
    transition("* => _", [
        query(
            ":leave",
            [
                style({
                    transform: "translateX(0) translateY(-50%)",
                    opacity: 1,
                }),
                animate(
                    "0.5s ease-in-out",
                    style({
                        transform: "translateX(100%) translateY(-50%)",
                        opacity: 0,
                    }),
                ),
            ],
            { optional: true },
        ),
    ]),
]);
