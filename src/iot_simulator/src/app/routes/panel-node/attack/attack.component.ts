import { Component } from "@angular/core";
import { FormControl, FormGroup, ReactiveFormsModule } from "@angular/forms";
import { PhantomAttackerAttack } from "@models/phantom-attacker";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideBomb, lucideCrosshair, lucideRadar, lucideShield } from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";

@Component({
    imports: [
        ReactiveFormsModule,
        BrnSelectModule,
        HlmButtonModule,
        HlmLabelModule,
        HlmSelectModule,
        NgIcon,
    ],
    providers: [provideIcons({ lucideBomb, lucideCrosshair, lucideRadar })],
    templateUrl: "attack.component.html",
    styles: `
        label:has(:checked) {
            @apply !border-primary;
        }
    `,
})
export class AttackComponent {
    protected form: FormGroup = new FormGroup({
        target: new FormControl(null),
        type: new FormControl(null),
    });
    protected phantomAttackerAttackToString = PhantomAttackerAttack.toString;
    protected phantomAttackerAttackToIcon = PhantomAttackerAttack.toIcon;
    protected phantomAttackerAttacks = PhantomAttackerAttack.Types;

    public constructor(
        private readonly _networkManager: NetworkManagerService,
    ) {}
}
