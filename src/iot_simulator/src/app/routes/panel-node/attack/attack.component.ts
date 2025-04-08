import { Component } from "@angular/core";
import { FormControl, FormGroup, ReactiveFormsModule } from "@angular/forms";
import { ActivatedRoute } from "@angular/router";
import { Device } from "@models/device";
import { Node } from "@models/node";
import { PhantomAttackerAttack } from "@models/phantom-attacker";
import { NgIcon, provideIcons } from "@ng-icons/core";
import {
    lucideBomb,
    lucideCrosshair,
    lucideRadar,
    lucideUnplug,
} from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { map } from "rxjs";

@Component({
    imports: [
        ReactiveFormsModule,
        BrnSelectModule,
        HlmButtonModule,
        HlmLabelModule,
        HlmSelectModule,
        NgIcon,
    ],
    providers: [
        provideIcons({
            lucideBomb,
            lucideCrosshair,
            lucideRadar,
            lucideUnplug,
        }),
    ],
    templateUrl: "attack.component.html",
    styles: `
        label:has(:checked) {
            @apply !border-primary;
        }
    `,
})
export class AttackComponent {
    private _node!: Node;
    protected form: FormGroup = new FormGroup({
        target: new FormControl(null),
        type: new FormControl(null),
    });
    protected get isConnected(): boolean {
        return this._node.connected;
    }
    protected get canConnect(): boolean {
        return this._networkManager.router !== undefined;
    }
    protected get connectedNodes(): Node[] {
        return this._networkManager
            .getConnectedNodes()
            .filter((node) => node.mac !== this._node.mac);
    }
    protected phantomAttackerAttackToString = PhantomAttackerAttack.toString;
    protected phantomAttackerAttackToIcon = PhantomAttackerAttack.toIcon;
    protected phantomAttackerAttacks = PhantomAttackerAttack.Types;

    public constructor(
        private readonly _route: ActivatedRoute,
        private readonly _networkManager: NetworkManagerService,
    ) {}

    public ngOnInit(): void {
        this._route
            .parent!.params.pipe(
                map(({ mac }) => this._networkManager.findByMac(mac)),
            )
            .subscribe((node) => (this._node = node));
    }

    public connect() {
        if (this.canConnect && this._node instanceof Device) {
            this._node.connect(this._networkManager.router!);
        }
    }
}
