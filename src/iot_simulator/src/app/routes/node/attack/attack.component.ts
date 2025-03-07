import { CommonModule } from "@angular/common";
import { Component } from "@angular/core";
import { FormControl, FormGroup, ReactiveFormsModule } from "@angular/forms";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideBomb, lucideCrosshair, lucideRadar } from "@ng-icons/lucide";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";

@Component({
    imports: [
        CommonModule,
        ReactiveFormsModule,
        BrnSelectModule,
        HlmButtonModule,
        HlmLabelModule,
        HlmSelectModule,
        NgIcon,
    ],
    providers: [provideIcons({ lucideBomb, lucideCrosshair, lucideRadar })],
    templateUrl: "attack.component.html",
})
export class AttackComponent {
    protected form: FormGroup = new FormGroup({
        target: new FormControl(""),
        attack_type: new FormControl(""),
    });
}
