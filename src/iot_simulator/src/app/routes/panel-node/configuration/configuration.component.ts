import { CommonModule } from "@angular/common";
import { Component, OnInit } from "@angular/core";
import { FormControl, FormGroup, ReactiveFormsModule } from "@angular/forms";
import { ActivatedRoute, Router } from "@angular/router";
import { HlmButtonModule } from "@components/ui/ui-button-helm/src";
import { CyberShieldModel } from "@models/cyber-shield";
import { Node, NodeType } from "@models/node";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideShield } from "@ng-icons/lucide";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmSwitchModule } from "@spartan-ng/ui-switch-helm";
import { map } from "rxjs";

@Component({
    imports: [
        CommonModule,
        ReactiveFormsModule,
        BrnSelectModule,
        HlmButtonModule,
        HlmInputModule,
        HlmLabelModule,
        HlmSelectModule,
        HlmSwitchModule,
        NgIcon,
    ],
    providers: [provideIcons({ lucideShield })],
    templateUrl: "configuration.component.html",
})
export class ConfigurationComponent implements OnInit {
    private _node!: Node;
    protected form: FormGroup = new FormGroup({
        name: new FormControl(""),
        ip: new FormControl(""),
        type: new FormControl(null),
        cyberShield: new FormControl(false),
        cyberShieldModel: new FormControl(null),
    });
    protected get type(): NodeType {
        return this._node.type;
    }
    protected typeToString = NodeType.toString;
    protected get nodeTypes() {
        return this.type === NodeType.ROUTER
            ? NodeType.RouterTypes
            : NodeType.DeviceTypes;
    }
    protected cyberShieldModelToString = CyberShieldModel.toString;
    protected cyberShieldModels = CyberShieldModel.Types;

    public constructor(
        private readonly _router: Router,
        private readonly _route: ActivatedRoute,
        private readonly _networkManager: NetworkManagerService,
    ) {}

    public ngOnInit(): void {
        this._route
            .parent!.params.pipe(
                map(({ mac }) => this._networkManager.findByMac(mac)),
            )
            .subscribe((node: Node) => {
                // Set values
                this._node = node;
                this.form.patchValue(
                    {
                        name: node.name,
                        ip: node.ip,
                        type: node.type,
                    },
                    { emitEvent: false },
                );

                // Enable/Disable fields
                this.form.get("ip")!.disable({ emitEvent: false });
                if (node.type === NodeType.ROUTER)
                    this.form.get("type")!.disable({ emitEvent: false });
                if (!false)
                    this.form
                        .get("cyberShieldModel")!
                        .disable({ emitEvent: false });
            });
        this.form.valueChanges.subscribe(
            ({ name, type, cyberShield, cyberShieldModel }) => {
                // Set values
                this._node.name = name;
                if (type) this._node.type = type;

                // Enable/Disable fields
                if (cyberShield)
                    this.form
                        .get("cyberShieldModel")!
                        .enable({ emitEvent: false });
                else
                    this.form
                        .get("cyberShieldModel")!
                        .disable({ emitEvent: false });
            },
        );
    }

    protected delete(): void {
        this._networkManager
            .deleteNode(this._node.mac)
            .then((deleted: boolean) => {
                if (deleted) this._router.navigate([""]);
            });
    }
}
