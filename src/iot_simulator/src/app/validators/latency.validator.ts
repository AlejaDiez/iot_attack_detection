import { AbstractControl, ValidationErrors, ValidatorFn } from "@angular/forms";

export function latencyValidator(): ValidatorFn {
    return (group: AbstractControl): ValidationErrors | null => {
        const latency: number = group.get("latency")!.value;
        const variation: number = group.get("latencyVariation")!.value;

        return latency >= variation ? null : { latencyVariation: true };
    };
}
