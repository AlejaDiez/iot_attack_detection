import { Component, OnInit } from "@angular/core";
import {
    FormControl,
    FormGroup,
    ReactiveFormsModule,
    Validators,
} from "@angular/forms";
import { ActivatedRoute } from "@angular/router";
import { Device } from "@models/device";
import { Node } from "@models/node";
import { Attacks, PhantomAttacker } from "@models/phantom-attacker";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideUnplug } from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { map } from "rxjs";
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
    ],
    providers: [provideIcons({ lucideUnplug })],
    templateUrl: "attack.component.html",
})
export class AttackComponent implements OnInit {
    private _node!: Node;
    protected form: FormGroup = new FormGroup({
        attack: new FormControl(null, [Validators.required]),
        target: new FormControl(null, [Validators.required]),
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
    protected get internalAttacks(): Attacks {
        return (this._node.generator as PhantomAttacker).internalAttacks;
    }
    protected get externalAttacks(): Attacks {
        return (this._node.generator as PhantomAttacker).externalAttacks;
    }

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

    protected connect() {
        if (this.canConnect && this._node instanceof Device) {
            this._node.connect(this._networkManager.router!);
        }
    }

    protected attack() {
        const { attack, target } = this.form.value;

        switch (attack.id) {
            case "dos":
                (this._node.generator as PhantomAttacker).dos(
                    target,
                    53,
                    100,
                    [32, 128],
                );
                break;
            default:
                if (typeof target === "string")
                    (this._node.generator as PhantomAttacker).attack(
                        attack,
                        target,
                    );
                else
                    (this._node.generator as PhantomAttacker).attack(
                        attack,
                        ...target,
                    );
                break;
        }
    }
}
