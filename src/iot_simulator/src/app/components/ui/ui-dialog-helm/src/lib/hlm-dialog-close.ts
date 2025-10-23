import { Directive, computed, input } from "@angular/core";
import { hlm } from "@spartan-ng/helm/utils";
import type { ClassValue } from "clsx";

@Directive({
    selector: "[hlmDialogClose],[brnDialogClose][hlm]",
    host: {
        "[class]": "_computedClass()",
    },
})
export class HlmDialogClose {
    public readonly userClass = input<ClassValue>("", { alias: "class" });

    protected readonly _computedClass = computed(() =>
        hlm(
            "data-[state=open]:bg-accent data-[state=open]:text-muted-foreground rounded-xs absolute right-4 top-4 flex items-center justify-center opacity-70 transition-opacity hover:opacity-100 focus:outline-none disabled:pointer-events-none [&_ng-icon]:shrink-0",
            this.userClass(),
        ),
    );
}
