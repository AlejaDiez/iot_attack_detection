import { Component, OnInit } from "@angular/core";
import {
    FormControl,
    FormGroup,
    ReactiveFormsModule,
    Validators,
} from "@angular/forms";
import { ActivatedRoute } from "@angular/router";
import { HlmLabelModule } from "@components/ui/ui-label-helm/src";
import { HlmMenuSeparatorComponent } from "@components/ui/ui-menu-helm/src";
import { Device } from "@models/device";
import { Commands } from "@models/flow-generator";
import { Node, NodeType } from "@models/node";
import { Packet } from "@models/packet";
import { NgIcon, provideIcons } from "@ng-icons/core";
import {
    lucideForward,
    lucideHourglass,
    lucideRepeat,
    lucideReply,
    lucideUnplug,
} from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { BrnTableModule } from "@spartan-ng/brain/table";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmTableModule } from "@spartan-ng/ui-table-helm";
import { HlmTabsModule } from "@spartan-ng/ui-tabs-helm";
import { map } from "rxjs";

@Component({
    imports: [
        ReactiveFormsModule,
        BrnSelectModule,
        BrnTableModule,
        HlmButtonModule,
        HlmInputModule,
        HlmLabelModule,
        HlmMenuSeparatorComponent,
        HlmSelectModule,
        HlmTableModule,
        HlmTabsModule,
        NgIcon,
    ],
    providers: [
        provideIcons({
            lucideForward,
            lucideHourglass,
            lucideUnplug,
            lucideRepeat,
            lucideReply,
        }),
    ],
    templateUrl: "network-traffic.component.html",
})
export class NetworkTrafficComponent implements OnInit {
    private _node!: Node;
    protected get ip(): string | undefined {
        return this._node.ip;
    }
    protected get type(): NodeType {
        return this._node.type;
    }
    protected get traffic(): Packet[] {
        return [...this._node.traffic.slice(-50)];
    }
    protected get isConnected(): boolean {
        return this._node.connected;
    }
    protected get canConnect(): boolean {
        return this._networkManager.router !== undefined;
    }
    protected get internalCommands(): Commands {
        return this._node.generator.internalCommands;
    }
    protected get externalCommands(): Commands {
        return this._node.generator.externalCommands;
    }
    protected get connectedNodes(): Node[] {
        return this._networkManager
            .getConnectedNodes()
            .filter((node) => node.mac !== this._node.mac);
    }
    protected readonly form: FormGroup = new FormGroup({
        command: new FormControl(null, [Validators.required]),
        target: new FormControl(null, [Validators.required]),
    });

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

    protected execute() {
        const { command, target } = this.form.value;

        switch (command.id) {
            case "ping":
                this._node.generator.ping(target);
                break;
            default:
                if (typeof target === "string")
                    this._node.generator.execute(command, target);
                else this._node.generator.execute(command, ...target);
                break;
        }
    }
}
