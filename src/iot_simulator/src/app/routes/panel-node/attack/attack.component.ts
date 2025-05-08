import { CommonModule } from "@angular/common";
import {
    Component,
    computed,
    effect,
    inject,
    input,
    InputSignal,
    model,
    ModelSignal,
    Signal,
} from "@angular/core";
import { FormsModule } from "@angular/forms";
import { Device } from "@models/device";
import { Node } from "@models/node";
import { Attack, PhantomAttacker } from "@models/phantom-attacker";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideUnplug } from "@ng-icons/lucide";
import { TranslateModule } from "@ngx-translate/core";
import { NetworkService } from "@services/network.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmMenuSeparatorComponent } from "../../../components/ui/ui-menu-helm/src/lib/hlm-menu-separator.component";

@Component({
    imports: [
        CommonModule,
        FormsModule,
        TranslateModule,
        BrnSelectModule,
        HlmButtonModule,
        HlmLabelModule,
        HlmMenuSeparatorComponent,
        HlmSelectModule,
        NgIcon,
        HlmMenuSeparatorComponent,
    ],
    providers: [provideIcons({ lucideUnplug })],
    templateUrl: "attack.component.html",
    host: { class: "flex flex-col gap-4" },
})
export class AttackComponent {
    public readonly network: NetworkService = inject(NetworkService);
    protected readonly node: InputSignal<Node> = input.required<Node>();
    protected readonly attack: ModelSignal<Attack | null> =
        model<Attack | null>(null);
    protected readonly target: ModelSignal<string | string[] | null> = model<
        string | string[] | null
    >(null);
    protected readonly multipleTargets: Signal<boolean> = computed(
        () => this.attack()?.multiple ?? false,
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

    public constructor() {
        effect(() => {
            if (this.multipleTargets())
                this.target.update((value) => {
                    if (value && !Array.isArray(value))
                        return [value as string];
                    return value;
                });
            else
                this.target.update((value) => {
                    if (value && Array.isArray(value)) return value[0];
                    return value;
                });
        });
    }

    protected connect() {
        (this.node() as Device).connect(this.network.router!);
    }

    protected execute() {
        switch (this.attack()!.id) {
            case "dos":
                (this.node().generator as PhantomAttacker).dos(
                    this.target() as string,
                    53,
                    [32, 128],
                    200,
                );
                break;
            default:
                (this.node().generator as PhantomAttacker).attack(
                    this.attack()!.id,
                    ...(Array.isArray(this.target()!)
                        ? this.target()!
                        : [this.target()!]),
                );
                break;
        }
    }
}
