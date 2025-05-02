import { Component, inject, input, InputSignal, Signal } from "@angular/core";
import { toSignal } from "@angular/core/rxjs-interop";
import {
    FormControl,
    FormGroup,
    ReactiveFormsModule,
    Validators,
} from "@angular/forms";
import { Device } from "@models/device";
import { Node } from "@models/node";
import { PhantomAttacker } from "@models/phantom-attacker";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideUnplug } from "@ng-icons/lucide";
import { TranslateModule } from "@ngx-translate/core";
import { NetworkService } from "@services/network.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { map, tap } from "rxjs";
import { HlmMenuSeparatorComponent } from "../../../components/ui/ui-menu-helm/src/lib/hlm-menu-separator.component";

@Component({
    imports: [
        ReactiveFormsModule,
        BrnSelectModule,
        HlmButtonModule,
        HlmLabelModule,
        HlmMenuSeparatorComponent,
        HlmSelectModule,
        NgIcon,
        HlmMenuSeparatorComponent,
        TranslateModule,
    ],
    providers: [provideIcons({ lucideUnplug })],
    templateUrl: "attack.component.html",
    host: { class: "flex flex-col gap-4" },
})
export class AttackComponent {
    public readonly network: NetworkService = inject(NetworkService);
    protected readonly node: InputSignal<Node> = input.required<Node>();
    protected readonly form: FormGroup = new FormGroup({
        attack: new FormControl(null, [Validators.required]),
        target: new FormControl(null, [Validators.required]),
    });
    protected readonly multiple: Signal<boolean> = toSignal(
        this.form.get("attack")!.valueChanges.pipe(
            tap(({ multiple }) => {
                const value = this.form.get("target")!.value;

                if (value) {
                    if (multiple && !Array.isArray(value))
                        this.form.get("target")!.setValue([value]);
                    else if (!multiple && Array.isArray(value))
                        this.form.get("target")!.setValue(value[0]);
                }
            }),
            map(({ multiple }) => multiple),
        ),
    );
    protected get canConnect(): boolean {
        return this.network.router !== undefined;
    }
    protected get attacks(): any[] {
        const internalAttacks = (this.node().generator as PhantomAttacker)
            .internalAttacks;
        const externalAttacks = (this.node().generator as PhantomAttacker)
            .externalAttacks;

        if (internalAttacks.length > 0 && externalAttacks.length > 0)
            return [...internalAttacks, "---", ...externalAttacks];
        return [...internalAttacks, ...externalAttacks];
    }
    protected get connectedNodes(): Node[] {
        return this.network.getConnectedNodes(this.node().mac);
    }

    protected connect() {
        (this.node() as Device).connect(this.network.router!);
    }

    protected attack() {
        const { attack, target } = this.form.value;

        switch (attack.id) {
            case "dos":
                (this.node().generator as PhantomAttacker).dos(
                    target,
                    53,
                    [32, 128],
                    200,
                );
                break;
            default:
                (this.node().generator as PhantomAttacker).attack(
                    attack,
                    ...(Array.isArray(target) ? target : [target]),
                );
                break;
        }
    }
}
